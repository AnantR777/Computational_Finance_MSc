import numpy as np
import matplotlib.pyplot as plt
import time

# do not run, takes too long

# Simulate an arithmetic Brownian motion
# dX(t) = mu*dt + sigma*dW(t)

# Define parameters and time grid
npaths = 20000  # number of paths
T = 1  # time horizon
nsteps = 2000  # number of time steps
dt = T/nsteps  # time step
t = np.linspace(0, T, nsteps + 1)  # observation times
mu, sigma = 0.12, 0.4  # model parameters

# Precompute a matrix of standard Gaussian random numbers
N = np.random.randn(nsteps, npaths)

# Only vector instructions
tic = time.time()
dX = mu * dt + sigma * np.sqrt(dt) * N
X = np.vstack([np.zeros(npaths), np.cumsum(dX, axis=0)])
toc = time.time()
print("Vectorized method took:", toc - tic)

# (a) Loop over time steps (see oup.m and cirp.m)
tic = time.time()
# Allocate and initialize all paths
Xa = np.zeros((nsteps + 1, npaths))
for i in range(nsteps):
    Xa[i + 1, :] = Xa[i, :] + mu * dt + sigma * np.sqrt(dt) * N[i, :]
toc = time.time()
print("Loop over time steps took:", toc - tic)
maxdiff = np.max(np.abs(X - Xa))
print("Max difference:", maxdiff)

# (b) Loop over the paths
tic = time.time()
Xb = np.zeros((nsteps + 1, npaths))
for j in range(npaths):
    dXb = mu * dt + sigma * np.sqrt(dt) * N[:, j]
    Xb[:, j] = np.hstack([0, np.cumsum(dXb)])
toc = time.time()
print("Loop over paths took:", toc - tic)
maxdiff = np.max(np.abs(X - Xb))
print("Max difference:", maxdiff)

# (c) 1. Two nested loops over time steps and paths
tic = time.time()
# Allocate and initialize all paths
Xc1 = np.zeros((nsteps + 1, npaths))
for j in range(npaths):
    for i in range(nsteps):
        Xc1[i + 1, j] = Xc1[i, j] + mu * dt + sigma * np.sqrt(dt) * N[i, j]
toc = time.time()
print("Two nested loops (time steps, paths) took:", toc - tic)
maxdiff = np.max(np.abs(X - Xc1))
print("Max difference:", maxdiff)

# (c) 2. Two nested loops over paths and time steps
tic = time.time()
# Allocate and initialize all paths
Xc2 = np.zeros((nsteps + 1, npaths))
for i in range(nsteps):
    for j in range(npaths):
        Xc2[i + 1, j] = Xc2[i, j] + mu * dt + sigma * np.sqrt(dt) * N[i, j]
toc = time.time()
print("Two nested loops (paths, time steps) took:", toc - tic)
maxdiff = np.max(np.abs(X - Xc2))
print("Max difference:", maxdiff)

# Compute the expected path
EX = mu * t

# Plot the expected, mean and sample path
plt.figure(1)
plt.plot(t, EX, 'k', t, np.mean(X, axis=1), ':k', t, X[:, ::1000], t, EX, 'k', t, np.mean(X, axis=1), ':k')
plt.legend(['Expected path', 'Mean path'])
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([-0.9, 1.1])
plt.title('Paths of an arithmetic Brownian motion dX(t) = \mu*dt + \sigma*dW(t)')
plt.savefig('abmpaths.pdf')

# Plot the probability density function at different times
fig, axes = plt.subplots(3, 1)
bins = 50

hist, bin_edges = np.histogram(X[40, :], bins=bins, density=True)
axes[0].bar(bin_edges[:-1], hist)
axes[0].set_ylabel('f_X(x,0.2)')
axes[0].set_xlim([-0.9, 1.1])

hist, bin_edges = np.histogram(X[100, :], bins=bins, density=True)
axes[1].bar(bin_edges[:-1], hist)
axes[1].set_xlim([-0.9, 1.1])
axes[1].set_ylabel('f_X(x,0.5)')

hist, bin_edges = np.histogram(X[-1, :], bins=bins, density=True)
axes[2].bar(bin_edges[:-1], hist)
axes[2].set_xlim([-0.9, 1.1])
axes[2].set_xlabel('x')
axes[2].set_ylabel('f_X(x,1)')

fig.suptitle('Probability density function of an arithmetic Brownian motion at different times')
plt.tight_layout()
plt.savefig('abmhist.pdf')

plt.show()