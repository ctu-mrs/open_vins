/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <limits>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// ---------------------------------------------------------------------------
// Structural (Type 1) null space for the camera+IMU system.
//
// The VIO system has 4 directions that are unobservable by physics for any
// motion, any feature set, and any trajectory:
//   n_1, n_2, n_3 — global X/Y/Z translation in world frame
//   n_4            — global yaw (rotation about gravity / world-Z axis)
//
// OpenVINS uses a Z-up world frame, so ĝ = [0, 0, 1]^T.
//
// For a global translation δp along world axis e, the affected state blocks are:
//   all world-frame positions (IMU, each clone, each global-representation SLAM feature)
// For a yaw δψ about ĝ, the affected blocks are:
//   all rotations (IMU + clones): add ĝ
//   all world-frame positions: add ĝ × p
//   IMU velocity: add ĝ × v
//
// Camera-IMU calibration (body-frame) and IMU biases (body-frame) are unaffected.
//
// Returns an n×4 matrix N whose columns span the structural null space.
// Columns are orthonormalized via thin QR so the leakage metric N^T P^{-1} N
// is scale-independent.
// ---------------------------------------------------------------------------
static Eigen::MatrixXd compute_vio_nullspace(const std::shared_ptr<State> &state) {
  const int n = state->max_covariance_size();
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, 4);

  // Gravity in world frame (OpenVINS Z-up → gravity points in -Z direction).
  // Using the actual gravity vector rather than the unit yaw axis keeps the notation
  // consistent between the position cross-product (skew(p)*g = p×g) and the rotation
  // term (R_GtoI * g), and is equivalent to the unit-vector form after QR normalisation.
  //const Eigen::Vector3d g(0.0, 0.0, -9.81);
  const Eigen::Vector3d g(0.0, 0.0, -1.0);

  // Helper: position cross gravity, matching user's skew(p)*g = p×g convention
  auto pos_cross_g = [&](const Eigen::Vector3d &p) -> Eigen::Vector3d { return skew_x(p) * g; };

  // IMU pose ---------------------------------------------------------------
  {
    const int q_id = state->_imu->q()->id();
    const int p_id = state->_imu->p()->id();
    const Eigen::Vector3d p = state->_imu->pos();
    // Translation null space: unit shift of each world-frame position
    N.block<3, 3>(p_id, 0) = Eigen::Matrix3d::Identity();
    // Yaw null space:
    //   rotation block — gravity expressed in IMU body frame (R_GtoI * g)
    //   position block  — p × g  (linear part of yaw rotation about origin)
    N.block<3, 1>(q_id, 3) = state->_imu->Rot() * g;
    N.block<3, 1>(p_id, 3) = pos_cross_g(p);
  }

  // IMU velocity: only the yaw direction affects it (v × g)
  {
    const int v_id = state->_imu->v()->id();
    N.block<3, 1>(v_id, 3) = pos_cross_g(state->_imu->vel());
  }

  // Camera clones ----------------------------------------------------------
  // Each clone has its own rotation R_GtoI_clone, so the body-frame yaw direction
  // (rotation block) is clone->Rot() * g, not the world-frame g.
  for (const auto &kv : state->_clones_IMU) {
    const auto &clone = kv.second;
    const int q_id = clone->q()->id();
    const int p_id = clone->p()->id();
    const Eigen::Vector3d p = clone->pos();
    N.block<3, 3>(p_id, 0) = Eigen::Matrix3d::Identity();
    N.block<3, 1>(q_id, 3) = clone->Rot() * g;
    N.block<3, 1>(p_id, 3) = pos_cross_g(p);
  }

  // Global-representation SLAM features -----------------------------------
  // Anchored features are expressed in a body frame, so global yaw/translation
  // act on them via their anchor clone (already handled above); skip them.
  for (const auto &kv : state->_features_SLAM) {
    const auto &feat = kv.second;
    if (LandmarkRepresentation::is_relative_representation(feat->_feat_representation))
      continue;
    const int f_id = feat->id();
    const Eigen::Vector3d p = feat->get_xyz(false);
    N.block<3, 3>(f_id, 0) = Eigen::Matrix3d::Identity();
    N.block<3, 1>(f_id, 3) = pos_cross_g(p);
  }

  // Orthonormalize via thin QR so the leakage metric N^T P^{-1} N is scale-independent
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(N);
  return qr.householderQ() * Eigen::MatrixXd::Identity(n, 4);
}

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Remove if we don't have enough
    if (ct_meas < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 4. Compute linear system for each feature, nullspace project, and reject
  unsigned int rejection_count = 0;
  unsigned int orig_feat_count = feature_vec.size();

  bool ns_diag_done = false;

  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Save H_x before projection for the first-feature null-space diagnostic
    Eigen::MatrixXd H_x_before_proj;
    if (!ns_diag_done) H_x_before_proj = H_x;

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    check_nullspace_projection_once(state, H_x_before_proj, H_x, Hx_order, ns_diag_done);

    /// Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      rejection_count++;
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  update_rejection_rate(rejection_count, orig_feat_count);

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 5. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  compute_hx_condition_number(state, Hx_big);

  verify_nullspace_numerical(state, Hx_big, Hx_order_big);

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = boost::posix_time::microsec_clock::local_time();

  compute_structural_nullspace_metrics(state);

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void UpdaterMSCKF::update_rejection_rate(unsigned int rejection_count, unsigned int orig_feat_count) {
  if (orig_feat_count > 0) {
    double curr_frame_rejection_rate = static_cast<double>(rejection_count) / orig_feat_count;
    _rejection_rate_history.push_back(curr_frame_rejection_rate);
    if (_rejection_rate_history.size() > REJECTION_RATE_WINDOW)
      _rejection_rate_history.pop_front();
    double sum = 0.0;
    for (double r : _rejection_rate_history)
      sum += r;
    _last_rejection_rate_sum = sum;
    _last_rejection_rate = sum / _rejection_rate_history.size();
  }
  PRINT_INFO("[MSCKF-UP]: Rejection rate: %.1f%% avg over %zu frames (this frame: %u/%u rejected)\n",
             100.0 * _last_rejection_rate, _rejection_rate_history.size(), rejection_count, orig_feat_count);
}

void UpdaterMSCKF::compute_hx_condition_number(const std::shared_ptr<State> &state, const Eigen::MatrixXd &Hx_big) {
  auto t_start = boost::posix_time::microsec_clock::local_time();

  // Prior-weight columns by sqrt(P_ii): singular values then reflect "how many prior sigmas
  // does this measurement constrain" rather than raw Jacobian scale differences.
  // κ is accumulated as a windowed geometric mean; σ_min as an arithmetic mean.
  Eigen::MatrixXd Hx_norm = Hx_big;
  Eigen::VectorXd prior_std = state->Cov().diagonal().cwiseSqrt();
  for (int i = 0; i < Hx_norm.cols(); i++) {
    if (prior_std(i) > 1e-10)
      Hx_norm.col(i) *= prior_std(i);
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hx_norm);
  const Eigen::VectorXd &sv = svd.singularValues();
  double sigma_max = sv(0);
  double sigma_min = sv(sv.size() - 1);
  bool degenerate = (sigma_min <= 1e-6 * sigma_max);

  _hx_log_kappa_history.push_back(degenerate ? std::numeric_limits<double>::infinity()
                                              : std::log(sigma_max / sigma_min));
  if (_hx_log_kappa_history.size() > REJECTION_RATE_WINDOW)
    _hx_log_kappa_history.pop_front();

  _hx_sigma_min_history.push_back(sigma_min);
  if (_hx_sigma_min_history.size() > REJECTION_RATE_WINDOW)
    _hx_sigma_min_history.pop_front();

  double log_sum = 0.0;
  int finite_count = 0;
  for (double lk : _hx_log_kappa_history) {
    if (std::isfinite(lk)) {
      log_sum += lk;
      finite_count++;
    }
  }
  _last_hx_condition_number = (finite_count > 0) ? std::exp(log_sum / finite_count)
                                                  : std::numeric_limits<double>::infinity();

  double sigma_sum = 0.0;
  for (double s : _hx_sigma_min_history)
    sigma_sum += s;
  _last_hx_sigma_min = sigma_sum / _hx_sigma_min_history.size();

  auto t_end = boost::posix_time::microsec_clock::local_time();
  PRINT_INFO("[MSCKF-UP]: Hx cond (windowed geom. mean): %.2e | sigma_min (windowed mean): %.2e | "
             "this frame: %s (sigma_max=%.2e, sigma_min=%.2e)\n",
             _last_hx_condition_number, _last_hx_sigma_min, degenerate ? "DEGENERATE" : "ok", sigma_max, sigma_min);
  PRINT_INFO("[MSCKF-TIMING]: Condition number metric: %.3f ms\n",
             (t_end - t_start).total_microseconds() * 1e-3);
}

void UpdaterMSCKF::verify_nullspace_numerical(const std::shared_ptr<State> &state, const Eigen::MatrixXd &Hx_big,
                                              const std::vector<std::shared_ptr<Type>> &Hx_order_big) {
  const Eigen::Vector3d g(0.0, 0.0, -1.0);

  // Map each variable pointer to its parent clone and role (q=true, p=false).
  // Restricting to clone columns removes IMU/calibration/bias variables that have
  // no Jacobian in the MSCKF H_x, avoiding spurious contributions to ||H*N_ana||.
  std::map<const Type *, std::pair<bool, std::shared_ptr<ov_type::PoseJPL>>> var_to_clone;
  for (const auto &kv : state->_clones_IMU) {
    var_to_clone[kv.second->q().get()] = {true, kv.second};
    var_to_clone[kv.second->p().get()] = {false, kv.second};
  }

  // Count clone-only columns present in Hx_order_big
  int local_dim = 0;
  for (const auto &var : Hx_order_big)
    if (var_to_clone.count(var.get()))
      local_dim += var->size();

  if (local_dim < 4) {
    PRINT_INFO("[MSCKF-NS-VERIFY]: No clone variables in Hx_big — skipping.\n");
    return;
  }

  // Build H_clones (clone columns of Hx_big) and N_clones (analytical clone null space)
  // simultaneously, both ordered by their appearance in Hx_order_big.
  Eigen::MatrixXd H_clones = Eigen::MatrixXd::Zero(Hx_big.rows(), local_dim);
  Eigen::MatrixXd N_clones = Eigen::MatrixXd::Zero(local_dim, 4);
  {
    int local_col = 0, hx_col = 0;
    for (const auto &var : Hx_order_big) {
      auto it = var_to_clone.find(var.get());
      if (it != var_to_clone.end()) {
        H_clones.block(0, local_col, Hx_big.rows(), var->size()) =
            Hx_big.block(0, hx_col, Hx_big.rows(), var->size());
        const auto &clone = it->second.second;
        if (it->second.first) {
          N_clones.block<3, 1>(local_col, 3) = clone->Rot() * g;
        } else {
          N_clones.block<3, 3>(local_col, 0) = Eigen::Matrix3d::Identity();
          N_clones.block<3, 1>(local_col, 3) = skew_x(clone->pos()) * g;
        }
        local_col += var->size();
      }
      hx_col += var->size();
    }
  }

  auto t_ana_start = boost::posix_time::microsec_clock::local_time();
  {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(N_clones);
    N_clones = qr.householderQ() * Eigen::MatrixXd::Identity(local_dim, 4);
  }
  auto t_ana_end = boost::posix_time::microsec_clock::local_time();

  double H_times_N_ana = (H_clones * N_clones).norm();

  // Numerical null space of H_clones — consistent with clone-only N_ana.
  auto t_num_start = boost::posix_time::microsec_clock::local_time();
  int null_dim_structural = 0;
  int null_dim_trivial = 0;
  Eigen::MatrixXd N_num;
  {
    Eigen::BDCSVD<Eigen::MatrixXd> svd_num(H_clones, Eigen::ComputeFullV);
    const Eigen::VectorXd &sv_num = svd_num.singularValues();
    int n_sv = (int)sv_num.size();
    double sv_max_num = (n_sv > 0) ? sv_num(0) : 1.0;
    for (int i = n_sv - 1; i >= 0; i--) {
      if (sv_num(i) < 1e-8 * sv_max_num)
        null_dim_structural++;
      else
        break;
    }
    null_dim_trivial = std::max(0, local_dim - n_sv);
    N_num = svd_num.matrixV().rightCols(null_dim_structural);
  }
  auto t_num_end = boost::posix_time::microsec_clock::local_time();

  double H_times_N_num = N_num.cols() > 0 ? (H_clones * N_num).norm() : 0.0;

  // Principal angle cosines: sv of N_clones^T * N_num; values near 1 = aligned subspaces
  std::string cosine_str = "N/A (structural null dim < 4)";
  if (N_num.cols() >= 4) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_cross(N_clones.transpose() * N_num);
    const auto &cos_a = svd_cross.singularValues();
    char buf[160];
    snprintf(buf, sizeof(buf), "[%.4f, %.4f, %.4f, %.4f]", cos_a(0), cos_a(1), cos_a(2), cos_a(3));
    cosine_str = buf;
  }
  PRINT_INFO("[MSCKF-NS-VERIFY]: clone_dim=%d | ||H_c*N_ana||_F=%.3e | ||H_c*N_num||_F=%.3e | "
             "Num NS: %d structural + %d trivial | Principal cosines: %s\n",
             local_dim, H_times_N_ana, H_times_N_num, null_dim_structural, null_dim_trivial, cosine_str.c_str());
  PRINT_INFO("[MSCKF-TIMING]: Clone N_ana build: %.3f ms | Numerical NS (BDCSVD): %.3f ms\n",
             (t_ana_end - t_ana_start).total_microseconds() * 1e-3,
             (t_num_end - t_num_start).total_microseconds() * 1e-3);
}

void UpdaterMSCKF::check_nullspace_projection_once(const std::shared_ptr<State> &state,
                                                   const Eigen::MatrixXd &H_x_before,
                                                   const Eigen::MatrixXd &H_x_after,
                                                   const std::vector<std::shared_ptr<Type>> &Hx_order,
                                                   bool &done) {
  if (done) return;
  done = true;

  const Eigen::Vector3d g(0.0, 0.0, -1.0);

  // Clone-only variable map
  std::map<const Type *, std::pair<bool, std::shared_ptr<ov_type::PoseJPL>>> var_to_clone;
  for (const auto &kv : state->_clones_IMU) {
    var_to_clone[kv.second->q().get()] = {true, kv.second};
    var_to_clone[kv.second->p().get()] = {false, kv.second};
  }

  int local_dim = 0;
  for (const auto &var : Hx_order)
    if (var_to_clone.count(var.get()))
      local_dim += var->size();

  if (local_dim < 4) {
    PRINT_INFO("[MSCKF-NS-DIAG]: No clone variables — skipping.\n");
    return;
  }

  // Build N_clones (analytical clone-only null space)
  Eigen::MatrixXd N_clones = Eigen::MatrixXd::Zero(local_dim, 4);
  {
    int local_col = 0;
    for (const auto &var : Hx_order) {
      auto it = var_to_clone.find(var.get());
      if (it != var_to_clone.end()) {
        const auto &clone = it->second.second;
        if (it->second.first) {
          N_clones.block<3, 1>(local_col, 3) = clone->Rot() * g;
        } else {
          N_clones.block<3, 3>(local_col, 0) = Eigen::Matrix3d::Identity();
          N_clones.block<3, 1>(local_col, 3) = skew_x(clone->pos()) * g;
        }
        local_col += var->size();
      }
    }
  }
  {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(N_clones);
    N_clones = qr.householderQ() * Eigen::MatrixXd::Identity(local_dim, 4);
  }

  // Extract clone-only columns from H_x (before and after)
  auto extract_clones = [&](const Eigen::MatrixXd &Hx_local) {
    Eigen::MatrixXd H_c = Eigen::MatrixXd::Zero(Hx_local.rows(), local_dim);
    int local_col = 0, hx_col = 0;
    for (const auto &var : Hx_order) {
      if (var_to_clone.count(var.get())) {
        H_c.block(0, local_col, Hx_local.rows(), var->size()) =
            Hx_local.block(0, hx_col, Hx_local.rows(), var->size());
        local_col += var->size();
      }
      hx_col += var->size();
    }
    return H_c;
  };

  double residual_before = (extract_clones(H_x_before) * N_clones).norm();
  double residual_after  = (extract_clones(H_x_after)  * N_clones).norm();
  PRINT_INFO("[MSCKF-NS-DIAG]: Before nullspace_project: ||H_c * N_clones||_F = %.6e | After: %.6e\n",
             residual_before, residual_after);
}

void UpdaterMSCKF::compute_structural_nullspace_metrics(const std::shared_ptr<State> &state) {
  const Eigen::MatrixXd &P = state->Cov();
  const Eigen::Vector3d g(0.0, 0.0, -1.0);

  // Collect clone variable ranges ordered by their covariance index.
  // Using sorted (id, size, is_q, clone) tuples ensures P_clones rows/cols
  // are laid out in the same order as the covariance matrix.
  auto t_ns_start = boost::posix_time::microsec_clock::local_time();
  using CloneVar = std::tuple<int, int, bool, std::shared_ptr<ov_type::PoseJPL>>;
  std::vector<CloneVar> clone_vars;
  for (const auto &kv : state->_clones_IMU) {
    int q_id = kv.second->q()->id(), p_id = kv.second->p()->id();
    if (q_id >= 0) clone_vars.push_back({q_id, 3, true, kv.second});
    if (p_id >= 0) clone_vars.push_back({p_id, 3, false, kv.second});
  }
  std::sort(clone_vars.begin(), clone_vars.end());

  int clone_dim = 0;
  for (const auto &cv : clone_vars) clone_dim += std::get<1>(cv);

  if (clone_dim < 4) {
    PRINT_INFO("[MSCKF-UP]: Not enough clone state for null-space metrics.\n");
    return;
  }

  // Extract P_clones and build N_clones in clone_vars order.
  // Full QR of N_clones materialises both the orthonormal null space (first 4 cols)
  // and its Euclidean complement N_perp (remaining cols) in one decomposition.
  Eigen::MatrixXd P_clones = Eigen::MatrixXd::Zero(clone_dim, clone_dim);
  Eigen::MatrixXd N_clones = Eigen::MatrixXd::Zero(clone_dim, 4);
  {
    int ri = 0;
    for (const auto &rvar : clone_vars) {
      int r_id = std::get<0>(rvar), r_sz = std::get<1>(rvar);
      int ci = 0;
      for (const auto &cvar : clone_vars) {
        int c_id = std::get<0>(cvar), c_sz = std::get<1>(cvar);
        P_clones.block(ri, ci, r_sz, c_sz) = P.block(r_id, c_id, r_sz, c_sz);
        ci += c_sz;
      }
      if (std::get<2>(rvar)) {
        N_clones.block<3, 1>(ri, 3) = std::get<3>(rvar)->Rot() * g;
      } else {
        N_clones.block<3, 3>(ri, 0) = Eigen::Matrix3d::Identity();
        N_clones.block<3, 1>(ri, 3) = skew_x(std::get<3>(rvar)->pos()) * g;
      }
      ri += r_sz;
    }
  }
  Eigen::HouseholderQR<Eigen::MatrixXd> qr_full(N_clones);
  Eigen::MatrixXd Q = qr_full.householderQ() * Eigen::MatrixXd::Identity(clone_dim, clone_dim);
  Eigen::MatrixXd N = Q.leftCols(4);
  Eigen::MatrixXd N_perp = Q.rightCols(clone_dim - 4);
  auto t_ns_end = boost::posix_time::microsec_clock::local_time();

  // LDLT factorization of P_clones — shared between leakage and observable-subspace metrics.
  auto t_ldlt_start = boost::posix_time::microsec_clock::local_time();
  Eigen::LDLT<Eigen::MatrixXd> ldlt(P_clones);
  auto t_ldlt_end = boost::posix_time::microsec_clock::local_time();

  // Null space leakage: ||N^T P_c^{-1} N||_F, normalised by the baseline at the first update.
  auto t_leak_start = boost::posix_time::microsec_clock::local_time();
  {
    Eigen::MatrixXd PinvN = ldlt.solve(N);
    Eigen::Matrix4d leak_mat = N.transpose() * PinvN;
    double raw_leakage = leak_mat.norm();
    if (_null_space_leakage_baseline < 0.0)
      _null_space_leakage_baseline = raw_leakage;
    _last_null_space_leakage = raw_leakage / _null_space_leakage_baseline;
  }
  auto t_leak_end = boost::posix_time::microsec_clock::local_time();

  // Observable-subspace condition number: σ_max / σ_min of N_⊥^T P_c^{-1} N_⊥.
  auto t_obs_start = boost::posix_time::microsec_clock::local_time();
  {
    Eigen::MatrixXd PinvNperp = ldlt.solve(N_perp);
    Eigen::MatrixXd Lambda_obs = N_perp.transpose() * PinvNperp;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Lambda_obs);
    const Eigen::VectorXd &sv = svd.singularValues();
    _last_obs_condition_number = (sv(sv.size() - 1) > 1e-10 * sv(0))
                                     ? sv(0) / sv(sv.size() - 1)
                                     : std::numeric_limits<double>::infinity();
  }
  auto t_obs_end = boost::posix_time::microsec_clock::local_time();

  PRINT_INFO("[MSCKF-UP]: clone_dim=%d | Null-space leakage: %.3e | Obs. subspace κ: %.2e\n",
             clone_dim, _last_null_space_leakage, _last_obs_condition_number);
  PRINT_INFO("[MSCKF-TIMING]: Clone N+P_c build: %.3f ms | LDLT(P_c): %.3f ms | "
             "Leakage metric: %.3f ms | Obs-κ metric: %.3f ms\n",
             (t_ns_end  - t_ns_start).total_microseconds()   * 1e-3,
             (t_ldlt_end - t_ldlt_start).total_microseconds() * 1e-3,
             (t_leak_end - t_leak_start).total_microseconds() * 1e-3,
             (t_obs_end  - t_obs_start).total_microseconds()  * 1e-3);
}
