import numpy as np
import matplotlib.pyplot as plt

# Brownian Bridge Parameters
npaths = 20000  # Number of paths
T = 1  # Time horizon
nsteps = 200  # Number of time steps
dt = T / nsteps  # Time step
t = np.arange(0, T + dt, dt)  # Observation times

# Monte Carlo Simulation for Brownian Bridge
W = np.cumsum(np.sqrt(dt) * np.random.randn(nsteps, npaths), axis=0)
time_factor = np.linspace(0, T, nsteps).reshape(-1, 1)
B = W - time_factor * W[-1, :] / T
B = np.concatenate([np.zeros((1, npaths)), B], axis=0)

# Fourier Transformation Section
N = 1000
dx = 0.1
x = dx * np.arange(-N/2, N/2)
dxi = (2 * np.pi) / (N * dx)
xi = dxi * np.arange(-N/2, N/2)

# Characteristic Function for Brownian Bridge
def char_func_BB(t, T, xi):
    """
    Characteristic function of a Brownian Bridge at time t.

    Parameters:
    t (float): The time at which the characteristic function is evaluated, where 0 <= t <= T.
    T (float): The total duration of the Brownian Bridge.
    xi (float): The parameter of the characteristic function.

    Returns:
    complex: The value of the characteristic function at time t for parameter xi.
    """
    variance = t * (T - t) / T
    return np.exp(-0.5 * variance * xi**2)

# Inverse Fourier Transformation at Different Points
def inverse_FT(t):
    char_func = char_func_BB(t, T, xi)
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(char_func))) / (N * dx)

# Characteristic Function at Different Time Points
Fa_point2 = char_func_BB(0.2, 1, xi)
Fa_point5 = char_func_BB(0.5, 1, xi)
Fa_T = char_func_BB(T, 1, xi)

# Plot Characteristic Function at Different Times
plt.figure(1, figsize=(12, 8))
plt.suptitle('Characteristic Function of Brownian Bridge at Different Times')

# At 0.2
plt.subplot(3, 1, 1)
plt.plot(xi, np.real(Fa_point2), 'r', label='Real')
plt.plot(xi, np.imag(Fa_point2), 'g', label='Imaginary')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi,0.2)')
plt.legend()

# At 0.5
plt.subplot(3, 1, 2)
plt.plot(xi, np.real(Fa_point5), 'r', label='Real')
plt.plot(xi, np.imag(Fa_point5), 'g', label='Imaginary')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi,0.5)')
plt.legend()

# At T
plt.subplot(3, 1, 3)
plt.plot(xi, np.real(Fa_T), 'r', label='Real')
plt.plot(xi, np.imag(Fa_T), 'g', label='Imaginary')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi,T)')
plt.legend()

# PDFs at Different Points VS Monte Carlo
fn_point2 = inverse_FT(0.2)
fn_point5 = inverse_FT(0.5)
fn_T = inverse_FT(T)

# Plot PDFs vs Monte Carlo Histograms
plt.figure(2, figsize=(12, 8))
plt.suptitle('Monte Carlo Histograms Overplotted With Char PDF at 0.2, 0.5, and T')

# At 0.2
plt.subplot(3, 1, 1)
plt.hist(B[int(0.2 * nsteps), :], bins=np.arange(-0.8, 1.21, 0.02), density=True)
plt.plot(x, np.real(fn_point2), 'r')
plt.xlim([-0.8, 1.2])
plt.ylabel('f_X(x,0.2)')

# At 0.5
plt.subplot(3, 1, 2)
plt.hist(B[int(0.5 * nsteps), :], bins=np.arange(-0.8, 1.21, 0.02), density=True)
plt.plot(x, np.real(fn_point5), 'r')
plt.xlim([-0.8, 1.2])
plt.ylabel('f_X(x,0.5)')

# At T
plt.subplot(3, 1, 3)
plt.hist(B[-1, :], bins=np.arange(-0.8, 1.21, 0.02), density=True)
plt.plot(x, np.real(fn_T), 'r')
plt.xlim([-0.8, 1.2])
plt.ylabel('f_X(x,T)')

plt.show()
