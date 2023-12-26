## Pricing of European options with the Heston stochastic volatility model
# dS = mu*S*dt + sqrt(V)*S*dW1 <=> dX = (mu-V/2)*dt + sqrt(V)*dW1
# dV = kappa*(Vbar-V)*dt + sigmaV*sqrt(V)*dW2
# dW1*dW2 = rho*dt
# The Black-Scholes-Merton price is given for reference
# dS = mu*S*dt + sigma*S*dW

import numpy as np
from scipy.stats import norm
import time
from payoff import payoff

# Contract parameters
T = 1  # maturity
K = 1.1  # strike price

# Market parameters
S0 = 1  # initial stock price
r = 0.02  # risk-free interest rate
q = 0  # dividend rate

# Model parameters
kappa = 3  # mean-reversion rate
Vbar = 0.1  # mean-reversion level
sigmaV = 0.25  # volatility of the volatility (vol of vol)
V0 = 0.08  # initial volatility
rho = -0.8  # correlation of W1 and W2
sigma = np.sqrt(Vbar)  # BS volatility

mu = r - q  # GBM drift parameter
Feller_ratio = 2 * kappa * Vbar / sigmaV**2  # for monitoring
print("Feller Ratio:", Feller_ratio)

# Fourier parameters
xwidth = 8  # width of the support in real space
ngrid = 2**8  # number of grid points
alphac = -6  # damping parameter for a call
alphap = 6  # damping parameter for a put

# Monte Carlo parameters; paths = nblocks*npaths
nsteps = 200  # number of time steps
dt = T / nsteps  # time step
t = np.arange(0, T + dt, dt)  # time grid
nblocks = 100  # number of blocks
npaths = 2000  # number of paths per block

# Analytical solution
start_time = time.time()
muABM = r - q - 0.5 * sigma**2  # drift coefficient of the arithmetic Brownian motion
d2 = (np.log(S0 / K) + muABM * T) / (sigma * np.sqrt(T))
d1 = d2 + sigma * np.sqrt(T)
Vca = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
Vpa = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
cputime_a = time.time() - start_time

print(f'{"Own analytical BS":20s} {Vca:15.10f} {Vpa:15.10f} {cputime_a:15.10f}')

# Fourier transform method will be similar but requires implementation of the characteristic function and numerical integration

# Fourier transform method
start_time_ft = time.time()

# Grids in real and Fourier space
N = ngrid // 2
b = xwidth / 2  # upper bound of the support in real space
dx = xwidth / ngrid  # grid step in real space
x = dx * np.arange(-N, N)  # grid in real space
dxi = np.pi / b  # Nyquist relation: grid step in Fourier space
xi = dxi * np.arange(-N, N)  # grid in Fourier space

# Characteristic function of X at t=0 for arithmetic Brownian motion
psi = lambda xi: 1j * muABM * xi - 0.5 * (sigma * xi)**2  # characteristic exponent
Psic = np.exp(psi(xi + 1j * alphac) * T)  # shifted characteristic function for a call
Psip = np.exp(psi(xi + 1j * alphap) * T)  # shifted characteristic function for a put


U = S0 * np.exp(b)
L = S0 * np.exp(-b)
o , gc, Gc = payoff(x, xi, alphac, K, L, U, S0, 1)  # Call
S, gp, Gp = payoff(x, xi, alphap, K, L, U, S0, -1)  # Put


# Discounted expected payoff computed with the Plancherel theorem
VcF = np.exp(-r * T) / (2 * np.pi) * np.trapz(np.real(Gc * np.conj(Psic))) * dxi  # call
VpF = np.exp(-r * T) / (2 * np.pi) * np.trapz(np.real(Gp * np.conj(Psip))) * dxi  # put



cputime_ft = time.time() - start_time_ft
print(f'{"Fourier":20s} {VcF:15.10f} {VpF:15.10f} {cputime_ft:15.10f}')

# Note: This section implements the Fourier transform method for option pricing.
# The characteristic function of log-price is defined and used in the calculation.


# Monte Carlo simulation
start_time_mc = time.time()
VcMCb = np.zeros(nblocks)
VpMCb = np.zeros(nblocks)
for j in range(nblocks):
    # Generate correlated standard Gaussian random numbers
    N1 = np.random.randn(nsteps, npaths)
    N2 = rho * N1 + np.sqrt(1 - rho**2) * np.random.randn(nsteps, npaths)
    # Allocate and initialise the paths for the variance and the log price
    # X = zeros(nsteps+1,npaths);

    V = np.vstack([V0 * np.ones(npaths), np.zeros((nsteps, npaths))])

    # Euler-Maruyama for the variance
    for i in range(nsteps):
        V[i + 1, :] = Vbar + (V[i, :] - Vbar) * np.exp(-kappa * dt) + \
                      np.sqrt((sigmaV**2 / kappa * (np.exp(-kappa * dt) - np.exp(-2 * kappa * dt)) * V[i, :] +
                               Vbar * sigmaV*2 / (2 * kappa) * (1 - np.exp(-kappa * dt))*2)) * N2[i, :]
        V[i + 1, :] = np.maximum(V[i + 1, :], np.zeros(npaths))  # avoid negative V

    # % Compute the increments of the arithmetic Brownian motion X = log(S/S0)
    dX = (mu - 0.5 * V[:-1, :]) * dt + np.sqrt(V[:-1, :]) * N1 * np.sqrt(dt)
    X = np.vstack([np.zeros(npaths), np.cumsum(dX, axis=0)])

    # % Transform X(T) to geometric Brownian motion S(T)
    S = S0 * np.exp(X[-1, :])

    # % Discounted expected payoff
    VcMCb[j] = np.exp(-r * T) * np.mean(np.maximum(S - K, 0))
    VpMCb[j] = np.exp(-r * T) * np.mean(np.maximum(K - S, 0))

VcMC = np.mean(VcMCb)
VpMC = np.mean(VpMCb)
scMC = np.sqrt(np.var(VcMCb) / nblocks)
spMC = np.sqrt(np.var(VpMCb) / nblocks)
cputime_MC = time.time() - start_time_mc

print(f'{"Monte Carlo":20s} {VcMC:15.10f} {VpMC:15.10f} {cputime_MC:15.10f}')
print(f'{"Monte Carlo stdev":20s} {scMC:15.10f} {spMC:15.10f}')