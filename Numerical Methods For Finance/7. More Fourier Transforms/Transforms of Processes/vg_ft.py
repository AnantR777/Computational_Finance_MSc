import numpy as np
import matplotlib.pyplot as plt


# -- Model Parameters -----
# Define Model Params
mu = 0.2
sigma = 0.3  # model parameters of the ABM
kappa = 0.05  # scale parameter of the gamma process = 1/beta = 1/rate
K = 1.1  # Strike Price
r = 0.05  # Interest Rate
S0 = 1
T = 1  # time horizonn


# -- VARIANCE GAMA MONTE CARLO SECTION -----
# Define parameters and time grid
npaths = 20000  # number of paths
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.linspace(0, T, nsteps+1)  # observation times

# Intrementing
# Compute the increments of the gamma process
dG = np.random.gamma(dt / kappa, kappa, (nsteps, npaths))
# Compute the increments of the ABM on the gamma random clock
dX = mu * dG + sigma * np.random.randn(nsteps, npaths) * np.sqrt(dG)
# Accumulate the increments to get our sample paths
X = np.vstack((np.zeros((1, npaths)), np.cumsum(dX, axis=0)))
# Exponentiate X to get the asset prices starting from 1 (as variance gamma is in log)
S = S0 * np.exp(X)

# -- FOURIER TRANSFORMATION SECTION -----
# -- Transformation Parameters --
N = 10000  # grid size
Dx = 0.01  # grid step in real space
Lx = N * Dx  # upper truncation limit in real space
Dxi = 2 * np.pi / Lx  # grid step in Fourier space
Lxi = N * Dxi  # upper truncation limit in Fourier space
x = Dx * np.arange(-N/2, N/2)  # grid in real space
xi = Dxi * np.arange(-N/2, N/2)  # grid in Fourier space

# Characteristic function - using parameters of the monte-carlo SDE equation.
def Fa(t): return (1 - 1j * kappa * mu * xi + 0.5 * kappa * (sigma * xi)**2)**(-t/kappa)
fa = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Fa(T)))) / Lx  # PDE at last timestep



# -- Visualisation of Real And Imaginary Characteristic function at 0.1, 0.5 and T -----
# -- Plot Characteristic Function at time T --
# Inverse Furior Transformation to get the function at different points
Fa_point2 = Fa(0.2) # Characteristic Function at 0.2
Fa_point5 = Fa(0.5) # Characteristic Function at 0.5
Fa_T = Fa(T) # Characteristic Function at T

plt.figure(1)
plt.suptitle('Characteristic Function At Different Times')

# Plot at 0.2 - Characteristic Function
plt.subplot(3, 1, 1)
plt.plot(xi, np.real(Fa_point2), 'r', label='Real Part')
plt.plot(xi, np.imag(Fa_point2), 'g', label='Imaginary Part')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi, 0.2)')
plt.legend(loc = 'upper right')

# Plot at 0.5 - Characteristic Function
plt.subplot(3, 1, 2)
plt.plot(xi, np.real(Fa_point5), 'r', label='Real Part')
plt.plot(xi, np.imag(Fa_point5), 'g', label='Imaginary Part')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi, 0.5)')
plt.legend(loc = 'upper right')

# Plot at T - Characteristic Function
plt.subplot(3, 1, 3)
plt.plot(xi, np.real(Fa_T), 'r', label='Real Part')  # Assuming Fa_T is defined for T
plt.plot(xi, np.imag(Fa_T), 'g', label='Imaginary Part')
plt.xlabel('xi')
plt.xlim([-20, 20])
plt.ylabel('Psi(xi, T)')
plt.legend(loc = 'upper right')

# Show the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# -- Visualisation of PDFs at Different Points VS Monte Carlo -----
# Inverse Furior Transformation to get the function at different points
fn_point2 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Fa(0.2)))) / Lx  # Function at 0.2
fn_point5 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Fa(0.5)))) / Lx  # Function at 0.5
fn_1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Fa(1)))) / Lx  # Function at 1

plt.figure(2)
plt.suptitle('VG MC Histograms Overplotted With Characteristic PDF at 0.2, 0.5, and T')

# Plot at 0.2 - real PDF
plt.subplot(3, 1, 1)
plt.hist(X[40, :], bins=np.arange(-0.8, 1.21, 0.02), density=True, label='Monte Carlo Simulation')
plt.plot(x, np.real(fn_point2), 'r', label='Characteristic PDF')
plt.xlim([-0.8, 1.2])
plt.ylim([0, 4])
plt.ylabel('$f_X(x, 0.2)$')
plt.legend()

# Plot at 0.5 - real PDF
plt.subplot(3, 1, 2)
plt.hist(X[100, :], bins=np.arange(-0.8, 1.21, 0.02), density=True, label='Monte Carlo Simulation')
plt.plot(x, np.real(fn_point5), 'r', label='Characteristic PDF')
plt.xlim([-0.8, 1.2])
plt.ylim([0, 4])
plt.ylabel('$f_X(x, 0.5)$')
plt.legend()

# Plot at 1 - real PDF
plt.subplot(3, 1, 3)
plt.hist(X[-1, :], bins=np.arange(-0.8, 1.21, 0.02), density=True, label='Monte Carlo Simulation')
plt.plot(x, np.real(fn_1), 'r', label='Characteristic PDF')
plt.xlim([-0.8, 1.2])
plt.ylim([0, 4])
plt.ylabel('$f_X(x, T)$')
plt.legend()

# Show the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()