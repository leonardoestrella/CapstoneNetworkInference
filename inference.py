import numpy as np

def infer_node_dynamics(phi_history, t, r, node_idx, method='backward'):
    """
    Infer the Fourier coefficients of both self-dynamics and pairwise interactions
    for a single oscillator in a network of phase oscillators.

    Parameters
    ----------
    phi_history : ndarray, shape (T, N)
        Time series of phase angles for N oscillators, sampled at T time points.
    t : ndarray, shape (T,)
        Array of time points corresponding to each row in phi_history.
    r : int
        Maximum Fourier mode to include for expansions.
    node_idx : int
        Index of the oscillator to be inferred (0 <= node_idx < N).
    method : {'backward', 'one_sided'}, optional
        Numerical differentiation method:
            - 'backward': (phi[i+1] - phi[i]) / Δt
            - 'one_sided': (3*phi[i+2] - 4*phi[i+1] + phi[i]) / (2*Δt)
        Defaults to 'backward'.

    Returns
    -------
    x_sol : ndarray, shape (1 + r + r^2*(N-1),)
        The least-squares solution vector. The first (1 + r) entries correspond
        to the “self” dynamics coefficients (constant term + r Fourier modes).
        The remaining r^2*(N-1) entries are the pairwise interaction coefficients
        for all other oscillators j != node_idx.
    residuals : ndarray
        Sum of squared residuals of the least-squares solution. 
        (Empty if the solution is a perfect fit or T <= # parameters.)
    """
    # -------------------------------------------------------------------------
    # 0) Validate inputs
    # -------------------------------------------------------------------------
    valid_methods = ['one_sided', 'backward']
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {valid_methods}.")

    # Check shapes
    if len(t) != phi_history.shape[0]:
        raise ValueError("Time array length must match the first dimension of phi_history.")
    if node_idx < 0 or node_idx >= phi_history.shape[1]:
        raise IndexError("node_idx is out of bounds for the second dimension of phi_history.")
    if r < 1:
        raise ValueError("r must be >= 1.")

    # Check we have enough data points for the chosen method
    T = phi_history.shape[0]
    if method == 'one_sided' and T < 3:
        raise ValueError("one_sided method requires at least 3 time points.")
    if method == 'backward' and T < 2:
        raise ValueError("backward method requires at least 2 time points.")

    # -------------------------------------------------------------------------
    # 1) Compute finite differences for dphi_i/dt (for the chosen method)
    # -------------------------------------------------------------------------
    if method == 'one_sided':
        # y_data shape -> (T-2,)
        delta_t = t[1:] - t[:-1]               # shape (T-1,)
        if len(delta_t) < 2:
            raise ValueError("Insufficient time points for 'one_sided' differencing.")
        # phi[i+2], phi[i+1], phi[i]
        y_data = (3.0 * phi_history[2:, node_idx]
                  - 4.0 * phi_history[1:-1, node_idx]
                  +        phi_history[:-2, node_idx]) / (2.0 * delta_t[1:])
        phi_used = phi_history[2:, :]         # shape (T-2, N)

    else:  # method == 'backward'
        # y_data shape -> (T-1,)
        delta_t = t[1:] - t[:-1]              # shape (T-1,)
        y_data = ((phi_history[1:, node_idx]
                  - phi_history[:-1, node_idx]) / delta_t)
        phi_used = phi_history[1:, :]         # shape (T-1, N)

    num_points = y_data.shape[0]
    N = phi_used.shape[1]

    # -------------------------------------------------------------------------
    # 2) Build expansions for the “self” node (i = node_idx)
    # -------------------------------------------------------------------------
    # a0 / sqrt(2π)
    A_const = np.ones((num_points, 1)) / np.sqrt(2.0 * np.pi)

    # For k=1..r, we have sin(k phi_i) + cos(k phi_i), scaled by 1/sqrt(π)
    modes = np.arange(1, r + 1)  # array([1, 2, ..., r])
    mat_i = np.outer(phi_used[:, node_idx], modes)  # shape (num_points, r)
    sin_i = np.sin(mat_i)
    cos_i = np.cos(mat_i)
    self_expansions = (sin_i + cos_i) / np.sqrt(np.pi)  # shape (num_points, r)

    # Combine into one partial design matrix for self: shape (num_points, 1 + r)
    A_self = np.hstack([A_const, self_expansions])

    # -------------------------------------------------------------------------
    # 3) Build expansions for each “other” node (j != i)
    #    bracket: [sin(l phi_j) + cos(l phi_j)] for l = 1..r
    # -------------------------------------------------------------------------
    mask_others = (np.arange(N) != node_idx)  
    phi_others = phi_used[:, mask_others]  # shape (num_points, N-1)

    # Expand in 3D for sin & cos
    mat_others_3d = phi_others[..., None] * modes  # shape (num_points, (N-1), r)
    sin_others_3d = np.sin(mat_others_3d)
    cos_others_3d = np.cos(mat_others_3d)

    expansions_others_3d = (sin_others_3d + cos_others_3d) / np.sqrt(np.pi)
    # Flatten across all j != i
    # shape from (num_points, (N-1), r) -> (num_points, (N-1)*r)
    expansions_others_2d = expansions_others_3d.reshape(num_points, -1)

    # -------------------------------------------------------------------------
    # 4) Form product of expansions_i with expansions_others => (num_points, r^2*(N-1))
    #    This is the pairwise term: [self_expansion_k * other_expansion_(k')]
    # -------------------------------------------------------------------------
    self_3d = self_expansions.reshape(num_points, r, 1)           # (num_points, r, 1)
    others_3d = expansions_others_2d.reshape(num_points, 1, r*(N-1))
    product_3d = self_3d * others_3d  # shape (num_points, r, r*(N-1))

    # Transpose so the middle axis lines up the “r*(N-1)” dimension nicely
    product_3d_alt = product_3d.transpose(0, 2, 1)  # (num_points, r*(N-1), r)

    # Flatten the last two dimensions => (num_points, r^2*(N-1))
    product_2d = product_3d_alt.reshape(num_points, -1)

    # -------------------------------------------------------------------------
    # 5) Combine design matrices: A = [ A_self | product_2d ]
    #    shape => (num_points, 1 + r + r^2*(N-1))
    # -------------------------------------------------------------------------
    A = np.hstack([A_self, product_2d])

    # -------------------------------------------------------------------------
    # 6) Solve for coefficients in least squares sense
    # -------------------------------------------------------------------------
    x_sol, residuals, rank, svals = np.linalg.lstsq(A, y_data, rcond=None)
    
    return x_sol, residuals