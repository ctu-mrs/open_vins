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

#ifndef OV_MSCKF_UPDATER_MSCKF_H
#define OV_MSCKF_UPDATER_MSCKF_H

#include <Eigen/Eigen>
#include <deque>
#include <memory>
#include <vector>

#include "feat/FeatureInitializerOptions.h"

#include "UpdaterOptions.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Will compute the system for our sparse features and update the filter.
 *
 * This class is responsible for computing the entire linear system for all features that are going to be used in an update.
 * This follows the original MSCKF, where we first triangulate features, we then nullspace project the feature Jacobian.
 * After this we compress all the measurements to have an efficient update and update the state.
 */
class UpdaterMSCKF {

public:
  /**
   * @brief Default constructor for our MSCKF updater
   *
   * Our updater has a feature initializer which we use to initialize features as needed.
   * Also the options allow for one to tune the different parameters for update.
   *
   * @param options Updater options (include measurement noise value)
   * @param feat_init_options Feature initializer options
   */
  UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief Given tracked features, this will try to use them to update the state.
   *
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);

  /// Returns the windowed-average chi2 rejection rate (fraction in [0,1])
  double get_rejection_rate() const { return _last_rejection_rate; }

  /// Returns the sum of per-frame rejection rates within the current window
  double get_rejection_rate_sum() const { return _last_rejection_rate_sum; }

  /// Returns the number of MSCKF updates currently in the sliding window
  size_t get_rejection_rate_window_size() const { return _rejection_rate_history.size(); }

  /// Returns the condition number of the compressed measurement Jacobian from the last MSCKF update.
  /// κ = σ_max / σ_min of the stacked H_x after null-space projection and QR compression.
  /// Large κ indicates near-degeneracy (instantaneous unobservability); returns inf when σ_min ≈ 0.
  double get_hx_condition_number() const { return _last_hx_condition_number; }

  /// Returns the minimum singular value of the compressed H_x from the last MSCKF update.
  /// Captures absolute measurement strength regardless of κ: both σ_max and σ_min can be small
  /// (weak measurements overall) while κ still looks healthy.
  double get_hx_sigma_min() const { return _last_hx_sigma_min; }

  /// Returns ||N^T P^{-1} N|| (Frobenius norm of the 4×4 null-space information matrix).
  /// For a consistent filter this should remain near zero; it grows when the filter
  /// falsely accumulates information about the structurally unobservable translation/yaw directions.
  double get_null_space_leakage() const { return _last_null_space_leakage; }

  /// Returns the condition number of N_⊥^T P^{-1} N_⊥ (observable-subspace information matrix).
  /// Large values mean some observable state directions are poorly informed relative to others.
  double get_obs_condition_number() const { return _last_obs_condition_number; }

protected:
  /// Updates the sliding-window chi2 rejection rate statistics.
  void update_rejection_rate(unsigned int rejection_count, unsigned int orig_feat_count);

  /// Computes the prior-weighted condition number and σ_min of the compressed H_x.
  /// Updates _last_hx_condition_number and _last_hx_sigma_min.
  void compute_hx_condition_number(const std::shared_ptr<State> &state, const Eigen::MatrixXd &Hx_big);

  /// Development-only: computes the numerical null space of H_full via full SVD and
  /// compares it against the analytical null space from compute_vio_nullspace.
  void verify_nullspace_numerical(const std::shared_ptr<State> &state, const Eigen::MatrixXd &Hx_big,
                                  const std::vector<std::shared_ptr<ov_type::Type>> &Hx_order_big);

  /// Development-only: for the first accepted feature (done=false), embeds H_x before and
  /// after nullspace_project_inplace into the full state space and prints ||H*N_ana||_F.
  /// Sets done=true after the first call so subsequent calls return immediately.
  void check_nullspace_projection_once(const std::shared_ptr<State> &state, const Eigen::MatrixXd &H_x_before,
                                       const Eigen::MatrixXd &H_x_after,
                                       const std::vector<std::shared_ptr<ov_type::Type>> &Hx_order, bool &done);

  /// Computes structural (Type 1) null space metrics on the post-update covariance:
  /// null-space leakage (N^T P^{-1} N) and observable-subspace condition number.
  /// Updates _last_null_space_leakage and _last_obs_condition_number.
  void compute_structural_nullspace_metrics(const std::shared_ptr<State> &state);

  /// Options used during update
  UpdaterOptions _options;

  /// Feature initializer class object
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;

  /// Sliding window of per-frame rejection rates (length <= REJECTION_RATE_WINDOW)
  static constexpr size_t REJECTION_RATE_WINDOW = 20;
  std::deque<double> _rejection_rate_history;

  /// Windowed-average rejection rate, sum, exposed via getters above
  double _last_rejection_rate = 0.0;
  double _last_rejection_rate_sum = 0.0;

  /// Sliding window of log(κ) for geometric-mean smoothing of the condition number.
  /// Stores +inf for degenerate frames (σ_min ≈ 0).
  std::deque<double> _hx_log_kappa_history;

  /// Sliding window of per-frame σ_min values (column-normalized H_x) for arithmetic-mean smoothing.
  std::deque<double> _hx_sigma_min_history;

  /// Windowed geometric mean of the condition number and windowed arithmetic mean of σ_min,
  /// exposed via getters above.
  double _last_hx_condition_number = 1.0;
  double _last_hx_sigma_min = 0.0;

  /// Structural (Type 1) null space metrics — updated after each EKF update.
  /// _null_space_leakage_baseline is captured on the first MSCKF update (<0 = not yet set).
  /// _last_null_space_leakage is the ratio current/baseline; stays near 1.0 for a consistent filter.
  double _null_space_leakage_baseline = -1.0;
  double _last_null_space_leakage = 1.0;
  double _last_obs_condition_number = 1.0;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_MSCKF_H
