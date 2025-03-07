import numpy as np
import matplotlib.pyplot as plt
import inference
import generate_synthetic_data
import generate_settings

# Generate synthetic data

t_vals, phi_vals = generate_synthetic_data.run_simulation(plot_data = True)

# Inferring the coefficient

r = 4 # number of terms in fourier series
N = phi_vals.shape[1]  # number of nodes

# backward inference
inferred_natural, inferred_couplings = inference.infer_all_nodes(phi_vals, t_vals, r, N, method = 'backward')
true_natural, true_couplings = generate_synthetic_data.prepare_comparison_trues()

# plotting natural and coupling

inference.plot_natural_frequencies(true_natural,inferred_natural)
inference.plot_coupling_strengths(true_couplings,inferred_couplings)