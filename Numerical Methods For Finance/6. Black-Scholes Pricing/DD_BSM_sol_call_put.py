import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# in lecture notes

# Parameters for drift-diffusion equation
mu = -0.05
D = 0.5
x, t = np.meshgrid(np.arange(-1, 1.02, 0.02), np.arange(0.1, 1.025, 0.025))
# 1-D arrays representing the coordinates of a grid used as ags
f = 1./(2*np.sqrt(np.pi*D*t)) * np.exp(-(x-mu*t)**2/(4*D*t)) # the solution
S0 = 1

# Plot the drift-diffusion equation solution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 111 means that there is only one subplot which occupies the entire figure
ax.plot_surface(x, t, f)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('f')
ax.set_title('Solution of the Fokker-Planck equation for arithmetic Brownian motion\nwith μ = -0.05, σ = 0.4')
ax.view_init(30, 24)
plt.savefig('abmfpe.png')

# Black-Scholes-Merton parameters
T = 1
K = 1.1
r = 0.05
q = 0.02
sigma = 0.4

# Create meshgrid for S (stock price) and t (time)
S, t = np.meshgrid(np.arange(0, 2.05, 0.05), np.arange(0, T+0.025, 0.025))
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*(T - t)) / (sigma*np.sqrt(T - t))
d2 = (np.log(S/K) + (r - q - 0.5*sigma**2)*(T - t)) / (sigma*np.sqrt(T - t))

# Calculate call option price
Vc = S*np.exp(-q*(T-t)) * norm.cdf(d1, 0, 1) - K*np.exp(-r*(T-t)) * norm.cdf(d2, 0, 1)
# cdf(x, loc=0, scale=1) for cdf mean 0 and var 1
Vc[-1,:] = np.maximum(S[-1,:] - K, 0)
#  selects the last row of the Vc matrix

# Plot call option price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, t, Vc)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('Vc')
ax.set_title('Call option value by Black-Scholes-Merton model')
ax.view_init(24, -30)
plt.savefig('bsc.png')

# Calculate put option price
Vp = K*np.exp(-r*(T-t)) * norm.cdf(-d2, 0, 1) - S*np.exp(-q*(T-t)) * norm.cdf(-d1, 0, 1)
Vp[-1,:] = np.maximum(K - S[-1,:], 0)

# Plot put option price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S, t, Vp)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('Vp')
ax.set_title('Put option value by Black-Scholes-Merton model')
ax.view_init(24, 30)
plt.savefig('bsp.png')

# Log price for Black-Scholes as a function of stock price
xa, t = np.meshgrid(np.arange(-1, 1.05, 0.05), np.arange(0, T+0.025, 0.025))
another_d1 = (xa - np.log(K/S0) + (r - q + 0.5*sigma**2)*(T - t)) / (sigma*np.sqrt(T - t))
# xa is log(S/S0). Therefore xa - log(K/S0) is log(S/K) giving the same d1 as before.
# Since xa is the log price, np.log(K/S0) adjusts the grid values to be
# centered around the log of the strike-to-initial-price ratio.
another_d2 = (xa - np.log(K/S0) + (r - q - 0.5*sigma**2)*(T - t)) / (sigma*np.sqrt(T - t))

# Calculate call option price as a function of log price
Vc_log = S0*(np.exp(xa - q*(T-t)) * norm.cdf(another_d1, 0, 1) - np.exp(np.log(K) - r*(T-t)) * norm.cdf(another_d2, 0, 1))
Vc_log[-1,:] = S0*np.maximum(np.exp(xa[-1,:]) - np.exp(np.log(K)), 0)

# Plot call option price as a function of log price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xa, t, Vc_log)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Vc')
ax.set_title('Call option value by Black-Scholes model')
ax.view_init(24, -30)
plt.savefig('bscx.png')

# Calculate put option price as a function of log price
Vp_log = S0*(np.exp(np.log(K) - r*(T-t)) * norm.cdf(-another_d2, 0, 1) - np.exp(xa - q*(T-t)) * norm.cdf(-another_d1, 0, 1))
Vp_log[-1,:] = S0*np.maximum(np.exp(np.log(K)) - np.exp(xa[-1,:]), 0)

# Plot put option price as a function of log price
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xa, t, Vp_log)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Vp')
ax.set_title('Put option value by Black-Scholes model')
ax.view_init(24, 30)
plt.savefig('bspx.png')

plt.show()