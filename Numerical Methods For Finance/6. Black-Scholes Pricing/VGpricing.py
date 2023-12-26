import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from parameters import parameters
from charfunction import charfunction
from kernel import kernel
from payoff import payoff
from figures_ft import figures_ft

# Contract parameters
T = 1  # maturity
K = 1.1  # strike price

# Market parameters
S0 = 1  # spot price
r = 0.05  # risk-free interest rate
q = 0.02  # dividend rate

# Model parameter
#sigma = 0.4  # volatility
#muABM = r - q - 0.5 * sigma ** 2

muABM = 0.2
sigma = 0.3

#gamma process parameter:
kappa=0.05 #scale parameter = 1/beta=1/rate

# Monte Carlo parameters; npaths = nblocks * nsample
nblocks = 2000  # number of blocks
nsample = 10000  # number of samples per block

# Fourier parameters
xwidth = 6  # width of the support in real space
ngrid = 2 ** 8  # number of grid points
alpha = -1  # damping factor for a call

#Controls
figures = 0


def payoff(x, xi, alpha, K, L, U, C, theta):
    # Scale
    S = C * np.exp(x)

    # Payoff
    # dampped payoff: g from slides:
    g = np.exp(alpha * x) * np.maximum(theta * (S - K), 0) * (
                (S >= L) & (S <= U))

    # Analytical Fourier transform of the payoff
    l = np.log(L / C)  # lower log barrier
    k = np.log(K / C)  # log strike
    u = np.log(U / C)  # upper log barrier

    # Integration bounds
    if theta == 1:  # call
        a = max(l, k)
        b = u
    else:  # put
        a = min(k, u)
        b = l

    # Fourier transform
    # equation: 3.26 in green, fusai, abrahams 2010
    # ghat from slides after applying fourier transform:

    xi2 = alpha + 1j * xi
    G = C * ((np.exp(b * (1 + xi2)) - np.exp(a * (1 + xi2))) / (1 + xi2) -
             (np.exp(k + b * xi2) - np.exp(k + a * xi2)) / xi2)

    # Eliminable discontinuities for xi = 0, otherwise 0/0 = NaN
    # // devision and rounds the the nearest int
    if alpha == 0:
        G[len(G) // 2] = C * (np.exp(b) - np.exp(a) - np.exp(k) * (b - a))
    elif alpha == -1:
        G[len(G) // 2] = C * (b - a + np.exp(k - b) - np.exp(k - a))

    # Plot to compare the analytical and numerical payoffs
    # gn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G))) / ((x[1] - x[0]) * len(x))
    # plt.figure()
    # plt.clf()
    # plt.plot(x, g, 'g', x, np.real(gn), 'r')
    # plt.xlabel('x')
    # plt.ylabel('g')
    # plt.legend(['analytical', 'numerical'])
    # if theta == 1:
    #    plt.title('Damped payoff function for a call option')
    # else:
    #    plt.title('Damped payoff function for a put option')
    # plt.show()

    return S, g, G


start_time = time.time()

# Fourier transform method
xwidth = 6  # width of the support in real space
ngrid = 2 ** 8  # number of grid points
alpha = -10  # damping factor for a call

# Grids in real and Fourier space
N = ngrid // 2
b = xwidth / 2  # upper bound of the support in real space
dx = xwidth / ngrid  # grid step in real space
x = dx * np.arange(-N, N)  # grid in real space
dxi = np.pi / b  # Nyquist relation: grid step in Fourier space
xi = dxi * np.arange(-N, N)  # grid in Fourier space

# Characteristic function at time T
# call
xia = xi + 1j * alpha
Psic = (1 - 1j * kappa * muABM * xia + 0.5 * kappa * (sigma * xia) ** 2) ** (
            -T / kappa)

# put
xia = xi - 1j * alpha
Psip = (1 - 1j * kappa * muABM * xia + 0.5 * kappa * (sigma * xia) ** 2) ** (
            -T / kappa)

# these function provide the characteristic function of 8 Levy processes:
# param = parameters(1,T,T,r,q); # set the parameters editing parameters.m
# [x,fc,xi,Psic] = kernel(ngrid,-b,b,param,alpha,0,1); # call
# [x,fp,xi,Psip] = kernel(ngrid,-b,b,param,-alpha,0,1); # put

# Fourier transform of the payoff
U = S0 * np.exp(b)
L = S0 * np.exp(-b)
_, gc, Gc = payoff(x, xi, alpha, K, L, U, S0, 1)  # call
S, gp, Gp = payoff(x, xi, -alpha, K, L, U, S0, -1)  # put

# Extra Figures:
# Call option
# figures_ft(S, x, xi, Psic, gc, Gc)
# Put option
# figures_ft(S, x, xi, Psip, gp, Gp)

# Discounted expected payoff computed with the Plancherel theorem
# final integral in pricing:
c = np.exp(-r * T) * np.real(np.fft.fftshift(
    np.fft.fft(np.fft.ifftshift(Gc * np.conj(Psic))))) / xwidth  # call
VcF = np.interp(S0, S, c, left=np.nan, right=np.nan)
p = np.exp(-r * T) * np.real(np.fft.fftshift(
    np.fft.fft(np.fft.ifftshift(Gp * np.conj(Psip))))) / xwidth  # put
VpF = np.interp(S0, S, p, left=np.nan, right=np.nan)
cputime_F = ((time.time() - start_time))
print('{:20s}{:<14.10f}{:<14.10f}{:<14.10f}'.format('Fourier', VcF, VpF,
                                                    cputime_F))

start_time = time.time()
VcMCb = np.zeros(nblocks)
VpMCb = np.zeros(nblocks)

for i in range(nblocks):
    # to change the process just change X:

    dG = np.random.gamma(T / kappa, kappa, (1, nsample))
    dX = muABM * dG + sigma * np.random.normal(size=(1, nsample)) * np.sqrt(dG)
    # Accumulate the increments
    S = S0 * np.exp(dX)  # Geometric Brownian motion
    # payoff
    VcMCb[i] = np.exp(-r * T) * np.mean(np.maximum(S - K, 0))  # Call option
    VpMCb[i] = np.exp(-r * T) * np.mean(np.maximum(K - S, 0))  # Put option

VcMC = np.mean(VcMCb)
VpMC = np.mean(VpMCb)
scMC = np.sqrt(np.var(VcMCb) / nblocks)
spMC = np.sqrt(np.var(VpMCb) / nblocks)
cputime_MC = ((time.time() - start_time))

print('{:20s}{:<14.10f}{:<14.10f}{:<14.10f}'.format('Monte Carlo', VcMC, VpMC,
                                                    cputime_MC))
print('{:20s}{:<14.10f}{:<14.10f}'.format('Monte Carlo stdev', scMC, spMC))

char_function = lambda t: (1 - 1j * xi * muABM * kappa + 0.5 * kappa * (sigma*xi) ** 2) ** (-t/kappa)
fn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift((char_function(T))))) / xwidth
nbins=np.arange(-1.5, 1.5, 0.02)
plt.hist(dX[0], bins=nbins, density=True,label="Monte Carlo - Sampled")
plt.plot(x, np.real(fn), 'r', label="Inverse of CF")
plt.ylabel(f'$f_X(x,T)$')
plt.legend()
plt.title('PDF of Variance Gamma @ time T')
plt.show()

# Step 1: Set Up Parameters
# Set the specific parameters for your model
param_obj = parameters(distr=3, T=1.0, dt=0.01, rf=0.05, q=0.02)  # Example parameters.
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