import numpy as np
import matplotlib.pyplot as plt

# Fourier Transformation in Arithmetic Brownian Motion
# -----------------------------------------------------------------------------
# This guide explores the simulation and analysis of Arithmetic Brownian Motion (ABM)
# and its Fourier Transformation. ABM is a fundamental concept in finance, used
# to model asset prices and other financial variables.

# ABM Equation: dX(t) = mu * dt + sigma * dW(t)
# where mu is the drift, sigma is the volatility, and dW(t) is the Wiener process increment.

# The guide consists of the parts:
# 1. Simulation of ABM paths
# 2. Fourier Transformation of ABM and comparison with simulated data

## 1. Simulation of ABM paths

# ---- Model Parameters for ABM ----

# Parameters for the Arithmetic Brownian Motion
muS = 0.2  # Drift coefficient
sigma = 0.55  # Volatility coefficient

# Define parameters and time grid for ABM
npaths = 20000  # Number of paths
T = 1  # Time horizon
nsteps = 200  # Number of time steps
dt = T/nsteps  # Time step
t = np.arange(0, T + dt, dt)  # Observation times

# Monte Carlo Simulation for ABM
# Compute the increments of the arithmetic Brownian motion
dW = (muS) * dt + sigma * np.random.randn(nsteps, npaths) * np.sqrt(dt)

# Accumulate the increments to form the ABM paths
X = np.concatenate([np.zeros((1, npaths)), np.cumsum(dW, axis=0)], axis=0)

## 2. Fourier Transformation of ABM and comparison with simulated data

# -- FOURIER TRANSFORMATION SECTION for ABM -----
# GRID IN REAL SPACE
N = 1000 # Number of grid points
dx = 0.1 # Grid step size in real space
upperx = N*dx # Upper truncation limit in real space
x = dx * np.arange(-N/2, N/2)  # grid in real space

# GRID IN FOURIER SPACE (Pulsation)
# Nyquist relation: dxdxi = 2pi/N
dxi = (2*np.pi)/(N*dx) # Grid step size in fourier space
upperxi = N*dxi # Upper truncation limit in fourier space
xi = dxi * np.arange(-N/2, N/2)  # grid in Fourier space

# GRID IN FOURIER SPACE (Frequency)
# Nyquist relation: dxdnu = 1/N
dnu = 1/(N*dx) # Grid step size in fourier space
uppernu = N*dnu # Upper truncation limit in fourier space
nu = dnu* np.arange(-N/2, N/2)  # Grid in fourier space


# Define the characteristic function for different times
def char_func_AB(t, xi):
    return np.exp(1j * xi * muS * t - 0.5 * (xi * sigma)**2 * t)

# Evaluate the characteristic function at specific time points
Fa_point2 = char_func_AB(0.2, xi)  # Characteristic function at time 0.2
Fa_point5 = char_func_AB(0.5, xi)  # Characteristic function at time 0.5
Fa_T = char_func_AB(T, xi)        # Characteristic function at time T

# Plot the characteristic function at different times
plt.figure(2)
plt.title('Characteristic Function At Different Times')

# Plot at 0.2 - Characteristic Function
plt.subplot(3, 1, 1)
plt.plot(xi, np.real(Fa_point2), 'r', label='Real Part')
plt.plot(xi, np.imag(Fa_point2), 'g', label='Imaginary Part')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi, 0.2)')
plt.legend()

# Plot at 0.5 - Characteristic Function
plt.subplot(3, 1, 2)
plt.plot(xi, np.real(Fa_point5), 'r', label='Real Part')
plt.plot(xi, np.imag(Fa_point5), 'g', label='Imaginary Part')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi, 0.5)')
plt.legend()

# Plot at T - Characteristic Function
plt.subplot(3, 1, 3)
plt.plot(xi, np.real(Fa_T), 'r', label='Real Part')
plt.plot(xi, np.imag(Fa_T), 'g', label='Imaginary Part')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel(f'Psi(xi, {T})')
plt.legend()

plt.show()


# Compute Fourier Transform using FFT
# fft and ifft require the origin to be in the beginning of the vector
# right now origin of char_func_AB at center, need to shift or origin, apply transform, shift back
f_X_AB = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func_AB(T, xi)))) / (N * dx)

# Repeat the process for frequency space
char_func1_AB = np.exp((1j * (2 * np.pi * nu) * muS - 0.5 * ((2 * np.pi * nu) * sigma)**2) * T)
f_X1_AB = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func1_AB))) / (N * dx)

# -- Visualization of PDFs at Different Points VS Monte Carlo -----

# Inverse Fourier Transformation to get the function at different points
# note here fft is actually the inverse transform since we defined it differently in notes
fn_point2_AB = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func_AB(0.2, xi)))) / upperx
fn_point5_AB = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func_AB(0.5, xi)))) / upperx
fn_T_AB = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func_AB(T, xi)))) / upperx
# normalise by truncation limit

# Plot PDFs at different points
plt.figure()
plt.title('Monte Carlo Histograms Overplotted With Char PDF at 0.2, 0.5 and T')

# Plot at 0.2
plt.subplot(3, 1, 1)
plt.hist(X[40, :], bins=np.arange(-0.8, 1.21, 0.02), density=True)
plt.plot(x, np.real(fn_point2_AB), 'r')
plt.xlim([-0.8, 1.2])
plt.ylabel('f_X(x,0.2)')

# Plot at 0.5
plt.subplot(3, 1, 2)
plt.hist(X[100, :], bins=np.arange(-0.8, 1.21, 0.02), density=True)
# bins chosen to be symmetric either side of mean of 0.2
plt.plot(x, np.real(fn_point5_AB), 'r')
plt.xlim([-0.8, 1.2])
plt.ylabel('f_X(x,0.5)')

# Plot at T
plt.subplot(3, 1, 3)
plt.hist(X[-1, :], bins=np.arange(-0.8, 1.21, 0.02), density=True)
plt.plot(x, np.real(fn_T_AB), 'r')
plt.xlim([-0.8, 1.2])
plt.ylabel('f_X(x,T)')

plt.show()