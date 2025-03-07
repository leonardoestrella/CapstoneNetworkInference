# Generating a Truncated Fourier basis

import numpy as np
import matplotlib.pyplot as plt
import generate_settings

def compute_derivative(phi, w, xi, eta, r):
    """
    Compute the time derivative for each oscillator using vectorized NumPy operations.
    
    Parameters:
        phi : ndarray, shape (N,)
            The current phases of the N oscillators.
        w : ndarray, shape (N,)
            The natural frequencies of the oscillators.
        xi : ndarray, shape (N, N, r)
            Coefficients for the sine terms in the Fourier series.
        eta : ndarray, shape (N, N, r)
            Coefficients for the cosine terms in the Fourier series.
        r : int
            Number of Fourier modes.
    
    Returns:
        phi_dot : ndarray, shape (N,)
            The computed time derivative for each oscillator.
    """
    N = phi.size
    # Compute the phase difference matrix: delta[i, j] = phi[i] - phi[j]
    delta = phi[:, None] - phi[None, :]  # Shape: (N, N)
    
    # Create Fourier mode multipliers (1-indexed)
    modes = np.arange(1, r + 1)  # Shape: (r,)
    
    # Compute sin and cos terms for all pairs (i, j) and all modes
    # Broadcasting: delta[..., None] has shape (N, N, 1) and modes is (r,) -> result is (N, N, r)
    sin_terms = np.sin(delta[..., None] * modes)
    cos_terms = np.cos(delta[..., None] * modes)

    # The functions are defined over an interval of 2 pi length, so the L = pi inside the
    # sine and cosine terms
    
    # Elementwise multiply by the coefficients and sum over the Fourier modes (axis 2)
    interaction_terms = (xi * sin_terms + eta * cos_terms).sum(axis=2)  # Shape: (N, N)
    
    # Exclude self-interaction by setting the diagonal to zero
    np.fill_diagonal(interaction_terms, 0)
    
    # Sum interactions over j for each oscillator
    total_interaction = interaction_terms.sum(axis=1)  # Shape: (N,)
    
    # Combine the intrinsic frequency and interaction terms
    phi_dot = w + total_interaction

    return phi_dot

def wrap_to_value(x,left_value,right_value):
    """
    Wraps the values to a value in a given interval [left_value, right_value).

    Parmeters:
        x: int, float, nparray
            values to be wrapped
        left_value: int, float
            left endpoint of the interval to wrap the value
        right_value: int, float
            right endpoint of the interval to wrap the value

    Returns:
        array
        Wrapped values
    """
    if left_value >= right_value:
        raise ValueError("left_value must be less than right_value")

    interval_size = right_value - left_value
    return left_value + (x - left_value) % interval_size

def rk2_step(phi, dt, w, xi, eta, r):
    """
    Perform a single RK2 integration step using
    the trapezoidal method
    
    Parameters:
        phi : ndarray, shape (N,)
            The current phases.
        dt : float
            Time step.
        w, xi, eta, r : parameters for the derivative function.
    
    Returns:
        Updated phi after one RK2 step. It turns the result into
        radians
    """
    k1 = compute_derivative(phi, w, xi, eta, r)
    k2 = compute_derivative(phi + dt/2 * k1, w, xi, eta, r)
    new_phi = phi + dt * k2
    return wrap_to_value(new_phi, -np.pi, np.pi)

def simulate_oscillators(N, r, T, dt, settings=None):
    """
    Simulate the dynamics of N oscillators over time T with time step dt.
    
    Parameters:
        N : int
            Number of oscillators.
        r : int
            Number of Fourier modes.
        T : float
            Total simulation time.
        dt : float
            Time step for integration.
        settigs : tuple or None
            If tuple, it must have initial state of the oscillators phi, the natural
            frequencies w, the coefficients of the sine xi, and of the cosine eta.
            Otherwise, these parameters are randomly chosen. 
    
    Returns:
        t_values : ndarray
            Array of time values.
        phi_history : ndarray
            History of oscillator phases (shape: [num_steps, N]).
    """
    num_steps = int(T / dt)

    if settings is None:
        # Initialize phases randomly between 0 and 2π
        phi = np.random.uniform(0, 2*np.pi, size=N)
        
        # Set natural frequencies (e.g., normally distributed around 1.0)
        w = np.random.normal(loc=1.0, scale=0.1, size=N)
        
        # Initialize random Fourier coefficients for interactions
        xi = np.random.rand(N, N, r)
        eta = np.random.rand(N, N, r)

    else: 
        phi, w, xi, eta = settings
    
    # Zero out self-interaction coefficients (i.e., when i == j)
    for i in range(N):
        xi[i, i, :] = 0
        eta[i, i, :] = 0

    t_values = np.linspace(0, T, num_steps)
    phi_history = np.zeros((num_steps, N))
    phi_history[0, :] = phi

    # Integrate using RK2
    for step in range(1, num_steps):
        phi = rk2_step(phi, dt, w, xi, eta, r)
        phi_history[step, :] = phi

    return t_values, phi_history


def data_for_inference(t_values, phi_history):
    """
    Prepares the synthetic data for inference. It returns the 
    data using the backward and one-sided numerical differentiation

    Parameters:
        t_values : ndarray
            Array of time values. (shape: [num_steps])
        phi_history : ndarray
            History of oscillator phases (shape: [num_steps, N]).
    
    Returns:
        delta_t_values : ndarray
            Array of time differences (shape: [num_steps-1])
        back_data : ndarray
            Numerical derivative using backward method (shape: [num_steps-1, N])
        one_sided_data : ndarray 
            Numerical derivative using one-sided method (shape: [num_steps-2, N])
    """

    delta_t = t_values[1:] - t_values[:-1]
    back_data = (phi_history[1:,:] - phi_history[:-1,:]) / delta_t[:, None]
    one_sided_data = (3*phi_history[2:] - 4 * phi_history[1:-1] + phi_history[:-2]) / (2*delta_t[1:, None])
    return delta_t, back_data, one_sided_data

def calculate_coupling_strength_per_node(w_i, xi_i, eta_i, node_idx):
    """
    Calculate and return the coupling strenght for the given settings
    for all the nodes in the model
    Parameters:
        w_i : float
            Natural frequency of the node i
        xi_i : ndarray (shape = (N,r))
            Sine coefficients for the node i
        eta_i : ndarray (shape = (N,r))
            Cosine coefficients for the node i
    Returns:
        array (shape (r+1,))
            Coupling strenghts for the given nodes i
            the first element is the natural frequency w_i,
            and each of the following are for the fourier
            modes
    """
    #w,xi,eta = generate_settings.settings
    coupling_self = np.array([np.sqrt(w_i**2)])
    coupling_others = np.sqrt(np.sum(xi_i**2 + eta_i**2, axis = 1))
    coupling_strength = np.concatenate([coupling_self, coupling_others])
    return coupling_strength

def calculate_coupling_strength():
    """
    Calculate and return the coupling strenght for all the nodes in the model
    Parameters:
        w : ndarray (shape = (N,))
            Natural frequencies of the nodes
        x : ndarray (shape = (N,N,r))
            Sine coefficients for the nodes
        eta : ndarray (shape = (N,N,r))
            Cosine coefficients for the nodes
    Returns:
        array (shape (N,r+1))
            Coupling strenghts for each node i
            the first element is the natural frequency w_i,
            and each of the following are for the fourier
            modes
    """
    phi, w,x,eta = generate_settings.settings
    coupling_strength = np.array([calculate_coupling_strength_per_node(w_i, xi_i, eta_i, node_idx) for node_idx, (w_i, xi_i, eta_i) in enumerate(zip(w, x, eta))])
    return coupling_strength

def prepare_comparison_trues():
    """
    Prepares the true dynamcis for comparison with the inferred values

    Parameters:
    Returns:
        array (shape (N,)), array (shape(N,N-1))
        The first array contains the true natural frequencies
        and the second the true coupling strengths
    """
    true_dynamics = calculate_coupling_strength()
    N = true_dynamics.shape[0]
    true_couplings = np.zeros((N,N-1))
    true_naturals = true_dynamics[:,0]
    for node_idx in range(N):
        true_couplings[node_idx,:] = np.delete(true_dynamics[node_idx,1:],node_idx)

    return true_naturals, true_couplings
def run_simulation(plot_data = False):
    """
    Run the simulation using the settings defined in generate_settings.

    Parameters:
        plot_data: bool (optional)
        If True, plots the simulation results. Defaults to False

    Returns:
        t (ndarray): Array of time points.
        phi_history (ndarray): History of oscillator phases.
    """
    r_synth = generate_settings.r_synth
    settings = generate_settings.settings
    N, T, dt = generate_settings.N, generate_settings.T, generate_settings.dt

    # Call simulate_oscillators with the appropriate settings
    t, phi_history = simulate_oscillators(N=len(settings[0]), r=r_synth, T=T, dt=dt, settings=settings)
    if plot_data:
        plot_simulation(t, phi_history)
    return t, phi_history

def plot_simulation(t, phi_history):
    """
    Plot the simulation results.

    Parameters:
        t (ndarray): Array of time points.
        phi_history (ndarray): History of oscillator phases.
    """
    plt.figure(figsize=(10, 6))
    for i in range(phi_history.shape[1]):
        plt.plot(t, phi_history[:, i], label=f'Oscillator {i+1}', linestyle = '--', 
                 marker = 'o', alpha = 0.7, markerfacecolor= 'none', markersize=4)
    plt.xlabel(r'Time ($t$, seconds)')
    plt.ylabel(r'Phase ($\phi$, radians)')
    plt.title('Synthetic Oscillatory Dynamics')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    t, phi_history = run_simulation()
    plot_simulation(t, phi_history)

# # Run the simulation and plot the results
# if __name__ == '__main__':

#     r = generate_settings.r
#     #phi, w, xi, eta = generate_settings.settings
#     settings = generate_settings.settings
#     N, T, dt = generate_settings.N, generate_settings.T, generate_settings.dt


#     t, phi_history = simulate_oscillators(N=len(settings[0]), r=r, T=T, dt=dt, settings = settings)
    
#     plt.figure(figsize=(10, 6))
#     for i in range(phi_history.shape[1]):
#         plt.plot(t, phi_history[:, i], label=f'Oscillator {i+1}')
#     plt.xlabel(r'Time ($t$, seconds)')
#     plt.ylabel(r'Phase ($\phi$, radians) ')
#     plt.title('Synthetic Oscillatory Dynamics')
#     plt.legend()
#     plt.show()
