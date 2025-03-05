import numpy as np    
import random as rd

#rd.seed(0)

np.random.seed(0)

N = 3
phi = np.random.rand(N) * 2 * np.pi # initial values
w = np.random.rand(N) * 2 * np.pi # natural frequencies
r = 3 # number of terms in the series
xi = (np.random.rand(N,N,r) + 1) / 2  # sine coefficients
eta = (np.random.rand(N,N,r) + 1) / 2  # cosine coefficients
settings = (phi,w,xi,eta)
dt = 0.01 # stepsize
T = 1.00 # time
