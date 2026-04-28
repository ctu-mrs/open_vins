# Sliding-window MSCKF chi2 rejection rate statistics.
# The window size is fixed at REJECTION_RATE_WINDOW (default 20 frames).

# Windowed average rejection rate in [0, 1].
# 0 = all features accepted, 1 = all features rejected.
float64 rejection_rate_avg

# Sum of per-frame rejection rates within the current window.
# Divide by window_size to recover rejection_rate_avg.
float64 rejection_rate_sum

# Number of MSCKF update frames currently in the sliding window.
# Ramps from 1 to REJECTION_RATE_WINDOW at startup, then stays constant.
uint32 window_size

# Condition number of the compressed measurement Jacobian H_x (σ_max / σ_min).
# Computed after null-space projection and QR compression, same update as rejection_rate_avg.
# Values near 1 indicate a well-conditioned (informative) measurement set.
# Large values indicate near-degeneracy due to poor motion or feature geometry.
# Infinity (encoded as +Inf) means at least one direction is completely unobservable this frame.
float64 hx_condition_number

# Minimum singular value of the compressed H_x.
# Captures absolute measurement strength independently of the condition number:
# both σ_max and σ_min can be small (weak measurements overall) while κ still looks healthy.
# Near zero means the current geometry/motion barely excites the observable subspace.
float64 hx_sigma_min

# Structural null space leakage: Frobenius norm of N^T * P^{-1} * N (4×4 matrix).
# N is the orthonormal basis for the 4 structurally unobservable directions (3 global
# translations + global yaw).  For a consistent filter this stays near zero; it grows
# when the filter falsely accumulates certainty about unobservable directions.
float64 null_space_leakage

# Observable subspace condition number: σ_max / σ_min of N_perp^T * P^{-1} * N_perp,
# where N_perp is the Euclidean orthogonal complement of the structural null space.
# Large values indicate some observable state directions are far better informed than
# others (imbalanced information distribution across the observable subspace).
float64 obs_condition_number
