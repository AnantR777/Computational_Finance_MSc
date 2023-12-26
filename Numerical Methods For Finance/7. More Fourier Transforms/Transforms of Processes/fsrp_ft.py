import numpy as np
import matplotlib.pyplot as plt

# Fourier Transformation in Feller Square Root (CIR) Process
# -----------------------------------------------------------------------------
# This guide focuses on simulating the Cox-Ingersoll-Ross (CIR) process,
# a type of Feller Square Root process, and analyzing it through Fourier Transformation.
# CIR process is often used in financial mathematics to model interest rates.

# CIR Process Equation: dX(t) = α(μ - X(t))dt + σ√X(t)dW(t)
# where α is the mean reversion speed, μ is the long-term mean level,
# σ is the volatility, and dW(t) is the Wiener process increment.

# The guide includes:
# 1. Simulation of the CIR process
# 2. Fourier Transformation and comparison with simulated data

# 1. SIMULATION OF THE CIR PROCESS
# -----------------------------------------------------------------------------
# Define simulation parameters and time grid
npaths = 20000  # Number of paths
T = 1  # Time horizon
nsteps = 200  # Number of time steps
dt = T / nsteps  # Time step
t = np.arange(0, T + dt, dt)  # Observation times

# Model parameters
alpha = 5
mu = 0.2
sigma = 0.4
S0 = 1  # Initial value
Feller_ratio = (2 * alpha * mu) / sigma**2  # Checking Feller condition

# Initialize paths
X = np.zeros((nsteps + 1, npaths))
X[0, :] = S0

# Generate random numbers
N = np.random.randn(nsteps, npaths)  # Standard normal random numbers

# Euler-Maruyama method for CIR process
for i in range(nsteps):
    X[i + 1, :] = X[i, :] + alpha * (mu - X[i, :]) * dt + \
                  sigma * np.sqrt(X[i, :] * dt) * N[i, :]
    X[i + 1, :] = np.maximum(X[i + 1, :], np.zeros(npaths))  # Ensure non-negativity

# Plotting the CIR process
plt.figure(1)
plt.plot(t, X[:, :10])
plt.xlabel("Time (t)")
plt.ylabel("X")
plt.title("CIR Process Simulation")

# 2. FOURIER TRANSFORMATION OF THE CIR PROCESS
# -----------------------------------------------------------------------------
# Fourier Transformation parameters
N = 2048  # Grid size
Dx = 0.01  # Grid step in real space
Lx = N * Dx  # Upper limit in real space
Dxi = 2 * np.pi / Lx  # Grid step in Fourier space
x = Dx * np.arange(-N/2, N/2)  # Real space grid
xi = Dxi * np.arange(-N/2, N/2)  # Fourier space grid

# CIR parameters for characteristic function
kappa = 5
theta = 0.2
sigma = 0.4
r0 = 1
T = 1

k = 4*kappa*theta/(sigma**2)


# Characteristic function for CIR process
def char_func_cir(t, u, v0, kappa, theta, sigma):
    # Parameters for the non-central chi-squared distribution
    ct = 2 * kappa / ((1 - np.exp(-kappa * t)) * sigma**2)
    k = 4 * kappa * theta / sigma**2
    lambda_t = 2 * ct * v0 * np.exp(-kappa * t)

    # Characteristic function of Y
    phi_Y_u = (1 - 1j * u / ct)**(-k / 2) * np.exp(1j * u * lambda_t / (2 * (ct - 1j * u)))

    return phi_Y_u

# Inverse Fourier Transformation
fn_point2 = np.fft.fftshift(np.fft.fft(
    np.fft.ifftshift(char_func_cir(0.2, xi, r0, kappa, theta, sigma)))) / Lx  # t=0.2
fn_point5 = np.fft.fftshift(np.fft.fft(
    np.fft.ifftshift(char_func_cir(0.5, xi, r0, kappa, theta, sigma)))) / Lx  # t=0.5
fn_1 = np.fft.fftshift(np.fft.fft(
    np.fft.ifftshift(char_func_cir(1, xi, r0, kappa, theta, sigma)))) / Lx  # t=1

# Plotting real probability density functions (PDFs) at different times
plt.figure(2)

# Plot at t=0.2
plt.subplot(3, 1, 1)
plt.hist(X[40, :], bins=np.arange(-0.8, 1.21, 0.02), density=True, label='Monte Carlo Simulation')
plt.plot(x, np.real(fn_point2), 'r',  label='Characteristic PDF')
plt.xlim([-0.8, 1.2])
plt.ylabel(f'$f_X(x,0.2)$')
plt.legend(loc = 'upper left')

# Plot at t=0.5
plt.subplot(3, 1, 2)
plt.hist(X[100, :], bins=np.arange(-0.8, 1.21, 0.02), density=True, label='Monte Carlo Simulation')
plt.plot(x, np.real(fn_point5), 'r',  label='Characteristic PDF')
plt.xlim([-0.8, 1.2])
plt.ylabel(f'$f_X(x,0.5)$')
plt.legend(loc = 'upper left')

# Plot at t=1
plt.subplot(3, 1, 3)
plt.hist(X[-1, :], bins=np.arange(-0.8, 1.21, 0.02), density=True, label='Monte Carlo Simulation')
plt.plot(x, np.real(fn_1), 'r',  label='Characteristic PDF')
plt.xlim([-0.8, 1.2])
plt.ylabel(f'$f_X(x,1)$')
plt.legend(loc = 'upper left')

plt.show()
