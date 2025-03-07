import numpy as np
import matplotlib.pyplot as plt

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
    x_sol : ndarray, shape (1 + rN^2,)
        The least-squares solution vector. The first element is the constant term
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
    fourier_bases = np.arange(1,r+1)

    # -------------------------------------------------------------------------
    # 2) Build the design matrix
    # -------------------------------------------------------------------------
    # a0 / sqrt(2π)
    A = np.ones((num_points, 1)) / np.sqrt(2.0 * np.pi)

    for mode1 in fourier_bases: # k
        for mode2 in fourier_bases: # l
            self_mode = phi_used[:,node_idx] * mode1
            P_kdx = (np.sin(self_mode) + np.cos(self_mode)) / np.sqrt(np.pi)    # Shape (t,)
            data_mode = phi_used * mode2
            P_ldx = (np.sin(data_mode) + np.cos(data_mode)) / np.sqrt(np.pi)    # Shape (t, N)

            O_ij = P_kdx[:, np.newaxis] * P_ldx
            A = np.hstack((A,O_ij))

    # -------------------------------------------------------------------------
    # 3) Solve for coefficients in least squares sense
    # from A with shape (t, rN**2 + 1 )
    # ------------------------------------------------------------------------
    x_sol, residuals, rank, svals = np.linalg.lstsq(A, y_data, rcond=None)
    
    # wrap the solutions, as they should be in the interval [0,2 pi] by how
    # we defined the range in which data are

    x_sol = x_sol % (2*np.pi) # sus, to be honest

    return x_sol, residuals

def infer_all_nodes(phi_vals,t_vals,r,N, method = 'backward'):
    """
    Infers the coupling strength of all the nodes in the network

    Parameters:
    ----------------
        phi_vals: array 
            Data to make the inference from
        t_vals: array
            time steps
        r: int
            Number of Fourier modes
        N: int
            Number of nodes to infer
        method: str
            'backward' or 'one_sided'
    Output:
    ----------------
        ndarray (shape (N,)), ndarray (shape (N,N-1))
            The first array contains the inferred natural frequencies
            and the second array contains the inferred coupling strenghts
    """
    

    inferred_couplings = np.zeros((N,N-1))
    inferred_natural = np.zeros(N)

    for node_idx in range(N):
        inferred_coefficient, error = infer_node_dynamics(phi_vals, t_vals, r, node_idx, method = method)
        natural, others = coeff_to_coupling_node(inferred_coefficient, node_idx, r, N)
        inferred_natural[node_idx] = natural
        inferred_couplings[node_idx,:] = others

    return inferred_natural, inferred_couplings

def coeff_to_coupling_node(inferred_coefficient,node_idx, r, N):
    """
    Convert the coefficients to coupling strenght of the fourier modes

    Parameters:
    ----------------
        inferred_coefficient: ndarray (shape: (r**2 * N + 1, ))
            inferred coefficients of the z_i
        node_idx: int
            index of the node to which the coefficients correspond (0 <= node_idx < N)
        r: int
            number of Fourier modes
        N: int
            Number of nodes in the network
    Output:
    ----------------
        float, ndarray
            The first float is the natural frequency as the constant term
            The second ndarray contains the coupling strengths of the fourier modes
            and has a shape(N-1) where its i-th element is the coupling strenght between
            node node_idx and i. 
        
    """
    constant_coefficient = inferred_coefficient[0]

    ordered_coefficient = inferred_coefficient[1:].reshape(r,r,N)
    # Each row i are the coefficients associated with the P_ij vector
    # each element ordered_coefficient[k,l,j] = b_{ij}^kl
    # for j = 1,2,3... r. 
    squared_coefficient = ordered_coefficient**2
    # The coupling stenght is a_ij = sum (b_{ij}^kl ** 2) for the kls

    coupling_strength = np.sqrt(np.sum(squared_coefficient, axis = (0,1))) 
    self_coupling_strenght = np.sqrt(coupling_strength[node_idx]**2+ constant_coefficient**2)
    return self_coupling_strenght, np.delete(coupling_strength, node_idx)

def plot_natural_frequencies(natural_frequencies,inferred_frequencies):
    """
    Plots the natural frequencies and the inferred frequencies
    Parameters:
        natural_frequencies: array (shape (N,)) of the true natural frequencies
        inferred_frequencies: array (shape (N,)) of the inferred frequencies
    """
    plt.scatter(natural_frequencies,inferred_frequencies, marker = 'o', facecolor = 'none', edgecolors='red')
    plt.xlabel("True natural frequency")
    plt.ylabel("Estimated natural frequency")
    plt.show();

def plot_coupling_strengths(coupling_strengths, inferred_coupling_strengths):
    """
    Plots the coupling strengths and the inferred coupling strengths
    Parameters:
        coupling_strengths: array (shape (N,N-1)) of the true coupling strengths
        inferred_coupling_strengths: array (shape (N,N-1)) of the inferred coupling strengths
    """
    plt.scatter(coupling_strengths.flatten(), inferred_coupling_strengths.flatten(), marker = 'o', facecolor = 'none', edgecolors='red')
    plt.xlabel("True Coupling Strength")
    plt.ylabel("Estimated Coupling Strength")
    plt.show();