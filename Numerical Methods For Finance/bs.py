import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fft, fftshift, ifftshift
from scipy.interpolate import interp1d
from scipy.stats import norm
from parameters import parameters
from charfunction import charfunction
from kernel import kernel
from payoff import payoff
from figures_ft import figures_ft
import py_vollib.black_scholes as bs
from scipy.integrate import trapz

# Option Pricing
# -----------------------------------------------------------------------------
# This script demonstrates different methods for pricing options: Analytical Black-Scholes model,
# Fourier transform methods, and Monte Carlo simulation.

# Define time grid and monte carlo parameters here: nsteps, dt, t, nblocks, npaths
# from script of process.
# remove nsample from below, copy nblocks


# Contract and Market Parameters
T = 1  # Maturity of the option
K = 1.1  # Strike price of the option
S0 = 1  # Current spot price of the underlying asset
r = 0.05  # Risk-free interest rate (annual)
q = 0.02  # Dividend yield (annual) of the underlying asset

# model parameters (change for diff model)
sigma = 0.4  # Volatility of the underlying asset

# Monte Carlo Simulation Parameters
nblocks = 2000  # Number of blocks for Monte Carlo simulation
nsample = 10000  # Number of samples per block

## Method 1: Analytical Black-Scholes Solution

# Analytical Black-Scholes Solution
# This section computes option prices using the Black-Scholes formula
start_time_analytical = time.time()
muGBM = r - q - 0.5 * sigma ** 2  # Drift term for the underlying asset
d2 = (np.log(S0 / K) + muGBM * T) / (sigma * np.sqrt(T))  # d2 term in Black-Scholes formula
d1 = d2 + sigma * np.sqrt(T)  # d1 term in Black-Scholes formula

# Calculating the call and put option prices analytically
Vca = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)  # Call option price
Vpa = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)  # Put option price
# Put-call parity: Vp + S0exp(-q*T) = Vc + Kexp(-rT)
end_time_analytical = time.time()
cputime_analytical  = end_time_analytical - start_time_analytical

# Analytical solution provided by Python's py_vollib
start_time_pyvollib = time.time()
VcaM = bs.black_scholes('c', S0, K, T, r, sigma)
VpaM = bs.black_scholes('p', S0, K, T, r, sigma)
end_time_pyvollib = time.time()
cputime_pyvollib = end_time_pyvollib - start_time_pyvollib

# Print the header
print(f"{'':>20}{'call':>14}{'put':>14}{'CPU_time/s':>14}")

# Print the values
print(f"{'BS analytical':>20}{Vca:>14.10f}{Vpa:>14.10f}{cputime_analytical:>14.10f}")
print(f"{'BS python':>20}{VcaM:>14.10f}{VcaM:>14.10f}{cputime_analytical:>14.10f}")

## Method 2: Fourier Transform

# Fourier Transform Method Parameters
start_time_fourier = time.time()
xwidth = 6  # Width of the support in real space (log price domain)
ngrid = 2 ** 8  # Number of grid points in the Fourier transform
alpha = -10  # Damping factor for the Fourier transform in call option pricing, for put would be -alpha

# Grids in Real and Fourier Space
N = ngrid // 2
b = xwidth / 2
dx = xwidth / ngrid  # Grid spacing in real space
x = dx * np.arange(-N, N)  # Real space grid (log price domain)
dxi = np.pi / b  # Grid spacing in Fourier space
xi = dxi * np.arange(-N, N)  # Fourier space grid


# Characteristic Function for Call and Put Options
# These functions are used in Fourier transform method for option pricing
xia = xi + 1j * alpha  # Shift for call option characteristic function
psi = 1j * muGBM * xia - 0.5 * (sigma * xia) ** 2
Psic = np.exp(psi * T)  # Characteristic function for call option
# change depending on process
xia = xi - 1j * alpha  # Shift for put option characteristic function
psi = 1j * muGBM * xia - 0.5 * (sigma * xia) ** 2
Psip = np.exp(psi * T)  # Characteristic function for put option

# Fourier Transform of Payoff
U = S0 * np.exp(b)  # Upper barrier for payoff
L = S0 * np.exp(-b)  # Lower barrier for payoff
_, gc, Gc = payoff(x, xi, alpha, K, L, U, S0, 1)  # Call payoff computation, 1 final argument indicates call
S, gp, Gp = payoff(x, xi, -alpha, K, L, U, S0, -1)  # Put payoff computation, -1 final argument indicates put


# Applying the Plancherel theorem for efficient option pricing using Fourier transform methods
# The Plancherel theorem allows us to work in the frequency domain to simplify computations

# Discount Expected Payoff Using Plancherel Theorem
# The theorem states that the integral of the square of a function is equal to the integral of the square of its Fourier transform
# This property is used here to transform the complex calculations of expected payoff into simpler computations in the frequency domain


c = (np.exp(-r * T) * np.real(fftshift(fft(ifftshift(Gc * np.conj(Psic))))) / xwidth)  # Call option pricing via FFT
# Explanation:
# Gc * np.conj(Psic): Multiplying the Fourier transform of the payoff function (Gc) with the complex conjugate of its characteristic function (Psic)
# fft(ifftshift(...)): Applying Fast Fourier Transform after rearranging zero-frequency components to the center
# fftshift(...): Rearranging the zero-frequency components back to the start after FFT
# np.real(...): Extracting the real part, as the final option price must be a real number
# np.exp(-r * T) / xwidth: Discounting the expected payoff at the risk-free rate (r) over time (T) and normalizing with the width of the grid (xwidth)
# This process results in computing the European call option price efficiently using the Fourier transform method

# Similar computation is done for the put option price using corresponding variables
p = (np.exp(-r * T) * np.real(fftshift(fft(ifftshift(Gp * np.conj(Psip))))) / xwidth)  # Put option pricing via FFT

VcF = np.interp(S0, S, c)  # Interpolating call option price at current spot price
#VcF = np.exp(-r * T) / (2 * np.pi) * trapz(np.real(Gc * np.conj(Psic))) * dxi / (2 * np.pi)
VpF = np.interp(S0, S, p)  # Interpolating put option price at current spot price
#VpF = np.exp(-r * T) / (2 * np.pi) * trapz(np.real(Gp * np.conj(Psic))) * dxi / (2 * np.pi)
end_time_fourier = time.time()
cputime_fourier = end_time_fourier - start_time_fourier
# Print the values
print(f"{'Fourier':>20}{VcF:>14.10f}{VpF:>14.10f}{cputime_fourier:>14.10f}")

# The np.interp() function estimates the option price at S0 based on surrounding computed values in S
# This step is essential because the Fourier transform method provides option prices for a range of asset prices in S,
# but we are specifically interested in the option price at the exact current market price of the asset, S0
# Linear interpolation is used to estimate this value, providing a practical option price for current market conditions


## Monte Carlo Simulation
start_time_mc = time.time()
# Monte Carlo Simulation for Option Pricing

# Initializing arrays to store option prices from each block
VcMcb = np.zeros(nblocks)  # Stores prices of call options for each block
VpMcb = np.zeros(nblocks)  # Stores prices of put options for each block

# Looping over each block to simulate option prices
for k in range(nblocks):
    # X represents the log returns of the underlying asset
    # It's calculated using the formula for geometric Brownian motion:
    # X = (muGBM * T) + (sigma * random shocks * sqrt(T))
    # muGBM * T is the drift component and the second term is the diffusion component
    # to price under different process, replace the below down to S = S0 * np.exp(X) with diff process
    # starting from generation of normal random number(s)
    # adapt accordingly for diff 1D process
    X = muGBM * T + sigma * np.random.randn(nsample) * np.sqrt(T)

    # Calculating simulated asset prices at maturity
    # S = S0 * exp(X) converts log returns to asset prices
    S = S0 * np.exp(X)

    # Calculating option payoffs for each simulated path and taking the mean
    # For call options: max(S - K, 0)
    # For put options: max(K - S, 0)
    # np.maximum() is used to apply the max function element-wise
    VcMcb[k] = np.exp(-r * T) * np.mean(np.maximum(S - K, 0))  # Call option payoff for block k
    VpMcb[k] = np.exp(-r * T) * np.mean(np.maximum(K - S, 0))  # Put option payoff for block k

# Computing the final option prices as the mean of the block prices
VcMC = np.mean(VcMcb)  # Final Monte Carlo price for call options
VpMC = np.mean(VpMcb)  # Final Monte Carlo price for put options

# Calculating the standard error of the simulation
# Standard error is the standard deviation of the block prices divided by the square root of the number of blocks
# This gives a measure of the accuracy of the Monte Carlo estimates
scMC = np.sqrt(np.var(VcMcb) / nblocks)  # Standard error for call option price
spMC = np.sqrt(np.var(VpMcb) / nblocks)  # Standard error for put option price
end_time_mc = time.time()
cputime_mc = end_time_mc - start_time_mc
# Print the values
print(f"{'Monte Carlo':>20}{VcMC:>14.10f}{VpMC:>14.10f}{cputime_mc:>14.10f}")
print(f"{'Monte Carlo dev':>20}{scMC:>14.10f}{spMC:>14.10f}")

## Plotting Analytical Solutions for Call and Put Options

# This section visualizes the option price as a function of the underlying asset price (S) and time to maturity (t).

# Create grids for plotting
St, t = np.meshgrid(np.arange(0, 2.05, 0.05), np.arange(0, T + 0.025, 0.025))
d2 = (np.log(St / K) + muGBM * (T - t)) / (sigma * np.sqrt(T - t))
d1 = d2 + sigma * np.sqrt(T - t)

# Plot the analytical solution for the Call option
Vc = St * np.exp(-q * (T - t)) * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
Vc[-1, :] = np.maximum(St[-1, :] - K, 0)  # Payoff at maturity
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(St, t, Vc)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')
ax.view_init(elev=24, azim=-30)
plt.title('Analytical Call Option Price')
#plt.savefig('bsc.png')

# Plot the analytical solution for the Put option
Vp = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - St * np.exp(-q * (T - t)) * norm.cdf(-d1)
Vp[-1, :] = np.maximum(K - St[-1, :], 0)  # Payoff at maturity
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(St, t, Vp)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')
ax.view_init(elev=24, azim=30)
plt.title('Analytical Put Option Price')
#plt.savefig('bsp.png')

# Plotting the analytical solution as a function of log price
# This section plots the option prices using log prices for better visualization.
k = np.log(K / S0)
xt, t = np.meshgrid(np.arange(-1, 1.05, 0.05), np.arange(0, T + 0.025, 0.025))
d2 = (xt - k + muGBM * (T - t)) / (sigma * np.sqrt(T - t))
d1 = d2 + sigma * np.sqrt(T - t)

# Call option with log price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xt, t, Vc)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('V')
ax.view_init(elev=24, azim=-30)
plt.title('Log Price Call Option')
#plt.savefig('bscx.png')

# Put option with log price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xt, t, Vp)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('V')
ax.view_init(elev=24, azim=30)
plt.title('Log Price Put Option')
#plt.savefig('bspx.png')
plt.show()

# Step 1: Set Up Parameters
# Set the specific parameters for your model
param_obj = parameters(distr=6, T=1.0, dt=0.01, rf=0.05, q=0.02)  # Example parameters.
# distr: Distribution type (e.g., 1 for Normal distribution).
# T: Time horizon or maturity.
# dt: Time increment.
# rf: Risk-free interest rate.
# q: Dividend yield.

# Step 2: Compute Characteristic Function

# Compute the characteristic function
char_func = charfunction(xi, param_obj)
# xi: Array of values in the Fourier space (to be defined).
# param_obj: Parameters object from Step 1.

# Step 3: Generate Kernel (PDF)
# Set the grid size and range for x

xmin, xmax = -5, 5  # Example range for x
# Compute the kernel or PDF
x, h, xi, H = kernel(ngrid, xmin, xmax, param_obj)
# ngrid: Number of grid points (to be defined).
# xmin and xmax: Range for the x grid.
# param_obj: Parameters object from Step 1.

# Step 4: Compute Payoffs
# Define parameters for payoff computation
K, L, U, C, theta = 100, 90, 110, 100, 1  # Example values for payoff computation
# K: Strike price of the option.
# L: Lower barrier.
# U: Upper barrier.
# C: Scaling factor, often set to the initial stock price.
# theta: Indicates call (1) or put (-1) option.

# Compute the payoffs
S, g, G = payoff(x, xi, 0, K, L, U, C, theta)
# x and xi: Grids in real and Fourier space.
# 0: Damping factor (alpha).
# K, L, U, C, theta: As defined above.

# Step 5: Visualize Results (using figures_ft.py)
figures_ft(S, x, xi, h, H, g, G)
# S: Scale, related to the asset price.
# x, xi: Grids in real and Fourier space.
# h, H: Kernel or PDF in real and Fourier space.
# g, G: Scaled payoffs in real and Fourier space.