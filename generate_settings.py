import numpy as np    
import random as rd

"""
# Uncomment and change these values to generate a random system
# Note: This is not fully developed and should include more details
# on how to generate the data from the original paper

np.random.seed(0) 
# Keep the seed constant, 

N = 3
phi = np.random.rand(N) * 2 * np.pi # initial values
w = np.random.rand(N) * 2 * np.pi # natural frequencies
r = 3 # number of terms in the series
xi = (np.random.rand(N,N,r) + 1) / 2  # sine coefficients
eta = (np.random.rand(N,N,r) + 1) / 2  # cosine coefficients
settings = (phi,w,xi,eta)
dt = 0.01 # stepsize
T = 10.00 # time"
"""
np.random.seed(5) 
N = 5
phi = np.random.uniform(-np.pi,np.pi, N)

w = np.random.uniform(-np.pi/2,np.pi/2,N)

#w = np.array([2.63 - np.pi/2 ,0.97 - np.pi/2, 2.4 - np.pi/2])

# bounds for the compact set
#left_bound, right_bound = 0, 12 # original paper - too strong!
left_bound, right_bound = 0,0.5
r_synth = 10 # number of terms in the series - not declared
all_coefficients = np.random.uniform(left_bound, right_bound, size = (N, 2*r_synth*N))

s = 0.45 # Sparsity
sparsity = np.random.choice([0, 1], size=(N, 2*r_synth*N), p=[1-s, s])
sparsified_coefficients = sparsity * all_coefficients

xi = sparsified_coefficients[:,:r_synth*N].reshape(N, N, r_synth)
eta = sparsified_coefficients[:,r_synth*N:].reshape(N, N, r_synth)

settings = (phi, w, xi, eta)

dt = 0.01
T = 12.00



