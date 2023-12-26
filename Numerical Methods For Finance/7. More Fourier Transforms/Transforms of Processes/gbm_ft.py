import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters and time grid
npaths = 20000  # number of paths
T = 1  # time horizon
nsteps = 200  # number of time steps
dt = T/nsteps  # time step
t = np.arange(0, T + dt, dt)  # observation times

mu = 0.2
sigma = 0.4
S0 = 1

## Monte Carlo
# Compute the increments of the arithmetic Brownian motion X = log(S/S0)==>S = S0 * np.exp(X)
dX = (mu-0.5*sigma**2)*dt + sigma*np.random.randn(npaths, nsteps)* np.sqrt(dt)
# Accumulate the increments
X = np.concatenate([np.zeros((npaths, 1)), np.cumsum(dX, axis=1)], axis=1)
# Transform to geometric Brownian motion
S = S0 * np.exp(X)

# -- FOURIER TRANSFORMATION SECTION for GBM -----
# GRID IN REAL SPACE
N = 1000 # Number of grid points
dx = 0.1 # Grid step size in real space
Lx = N*dx # Upper truncation limit in real space
x = dx * np.arange(-N/2, N/2)  # grid in real space

# GRID IN FOURIER SPACE (Pulsation)
dxi = (2*np.pi)/(N*dx) #Grid step size in fourier space
Lxi = N*dxi # Upper truncation limit in fourier space
xi = dxi * np.arange(-N/2, N/2)  # grid in Fourier space



# Characteristic function
muABM=(mu-0.5*sigma**2)
CF = lambda t: np.exp(1j*xi*muABM*t - 0.5*(xi**2)*(sigma**2)*t)

print(S[:, -1])

#1/nsteps*20=1/200*20=0.1
plt.figure(1)
plt.subplot(3, 1, 1)
plt.hist(np.log(S[:, 20]), bins=np.arange(-1, 1.3, 0.02), density=True, label = 'MC Simulation')
fn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CF(0.10)))) / Lx
plt.plot(x, np.real(fn), 'r', label='IFFT Result')
plt.xlim([-1, 1.3])
plt.ylim([0, 4])
plt.ylabel('f_X(x,0.10)')
plt.title("PDF of the GBM Model at t = 0.2")
plt.legend()

plt.subplot(3, 1, 2)
plt.hist(np.log(S[:, 80]), bins=np.arange(-1, 1.3, 0.02), density=True, label = 'MC Simulation')
fn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CF(0.4)))) / Lx
plt.plot(x, np.real(fn), 'r', label='IFFT Result')
plt.xlim([-1, 1.3])
plt.ylim([0, 4])
plt.ylabel('f_X(x,0.4)')
plt.title("PDF of the GBM Model at t = 0.4")
plt.legend()


plt.subplot(3, 1, 3)
plt.hist(np.log(S[:, -1]), bins=np.arange(-1, 1.3, 0.02), density=True, label = 'MC Simulation')
fn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CF(1)))) / Lx
plt.plot(x, np.real(fn), 'r', label='IFFT Result')
plt.xlim([-1, 1.3])
plt.ylim([0, 3.5])
plt.ylabel('f_X(x,1)')
plt.title("PDF of the GBM Model at t = 1")
plt.xlabel("Asset Price")
plt.legend()

# Show the plots
plt.tight_layout()

D = sigma**2/2 # diffusion coefficient
x, tt = np.meshgrid(np.arange(-1, 1.02, 0.02), np.arange(0.1, 1.025, 0.025))
f = 1./(2*np.sqrt(np.pi*D*tt)) * np.exp(-(x-mu*tt)**2./(4*D*tt))

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, tt, f, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f_X')
ax.set_title('Geometric Brownian motion: solution of the Fokker-Planck equation with mu=-0.05, sigma=0.4')
plt.show()

