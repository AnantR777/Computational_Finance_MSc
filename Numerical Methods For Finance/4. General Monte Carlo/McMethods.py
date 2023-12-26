import numpy as np
import matplotlib.pyplot as plt

# The below is a guide for implementing a Monte Carlo simulation for SDEs
# in python in a variety of different ways.
# They should all ultimately lead to the same outcome and so the purpose of
# the below is to show the different ways it can be written.
# -------------------------------------------------
# Simpler Stochastic Differential Equations
# Example: Arithmetic Brownian Motion
# -------------------------------------------------

# By simpler we mean they can be calculated using the cumsum function,
# rather than requiring an iterative approach.

# ABM forumla:
# dY(t) = mu*dt + sigma*dW(t)

# Parameters for Arithmetic Brownian Motion (ABM)
npaths = 20000  # Number of paths
T = 1  # Time horizon
nsteps = 200  # Number of time steps
dt = T / nsteps  # Time step size
t = np.linspace(0, T, nsteps + 1)  # Time grid
mu = 0.12  # Drift coefficient
sigma = 0.4  # Diffusion coefficient

# Create an [npaths,nsteps] matrix to simulate the value at each time step along each path
# Generate random increments
dY = mu * dt + sigma * np.sqrt(dt) * np.random.randn(npaths, nsteps)

## Method 1: Rows x Columns - Cumsum function
# We need to cumulatively sum the values over the time steps to get each path

Y1 = np.hstack((np.zeros((npaths, 1)), np.cumsum(dY, axis=1)))

# Method 2: Rows x Columns - For loop over nsteps
Y2 = np.zeros((npaths, nsteps + 1))
for i in range(nsteps):
    Y2[:, i + 1] = Y2[:, i] + dY[:, i]

# Method 3: Rows x Columns - For loop over npaths
Y3 = np.zeros((npaths, nsteps + 1))
for i in range(npaths):
    Y3[i, 1:] = np.cumsum(dY[i, :])

# Method 4: Rows x Columns - 2 For loops over nsteps then npaths
Y4 = np.zeros((npaths, nsteps + 1))
for i in range(npaths):
    for j in range(nsteps):
        Y4[i, j + 1] = Y4[i, j] + dY[i, j]

# Method 5: Rows x Columns - 2 For loops over npaths then nsteps
Y5 = np.zeros((npaths, nsteps + 1))
for j in range(nsteps):
    for i in range(npaths):
        Y5[i, j + 1] = Y5[i, j] + dY[i, j]

## Graphical Test of each method
# Only one line should appear as they should all be the same, hence stacked
# on top of each other.

plt.figure()
plt.plot(t, Y1[0, :], 'r', label='Method 1')
plt.plot(t, Y2[0, :], 'k', label='Method 2')
plt.plot(t, Y3[0, :], 'c', label='Method 3')
plt.plot(t, Y4[0, :], 'm', label='Method 4')
plt.plot(t, Y5[0, :], 'b', label='Method 5')
plt.title('Monte Carlo Simulation of ABM - Comparison of Methods')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()