import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the parameters
mu = 0.2  # mean
sigma = 0.1  # standard deviation
a = -0.4  # left truncation
b = 0.8  # right truncation
ngrid = 200  # number of grid intervals
nsample = 10**6  # number of random samples

# Define the grid
x = np.linspace(a, b, ngrid + 1)
deltax = x[1] - x[0]  # grid step, note equally spaced so just pick first diff

# Compute the PDF and CDF
f1 = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) / sigma)**2 / 2)  # Manual PDF
f = norm.pdf(x, mu, sigma)  # PDF using scipy
F = norm.cdf(x, mu, sigma)  # CDF using scipy

# Plot the PDF and CDF
plt.figure(1)
plt.plot(x, f1, 'r', label='PDF Manual')
plt.plot(x, f, 'bo', label='PDF Scipy')
plt.plot(x, F, 'g', label='CDF Scipy')
plt.xlim([a, b])
plt.xlabel('x')
plt.legend()
plt.title('Normal distribution with μ = 0.2 and σ = 0.1')
plt.savefig('normal.png')

# Sample the normal distribution
# U = np.random.rand(nsample)  # standard uniform random numbers
# Convert uniform random numbers to normal distribution (inverse transform method)
# X = mu + sigma * norm.ppf(U) # method 1a
# X = norm.ppf(U, loc=mu, scale=sigma) # method 1b
# X = mu + sigma * np.random.randn(nsample) # method 2 without using inverse cdf
X = np.random.normal(mu, sigma, nsample) # method 3, getting normal directly

# Plot histogram with normal fit
plt.figure(2)
plt.hist(X, bins=ngrid, density=True, alpha=0.6)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f')
plt.legend(['Sampled', 'Normal fit'])
plt.title('Normal distribution with μ = 0.2 and σ = 0.1')

# Plot normalized frequency vs. exact pdf
plt.figure(3)
h, bins = np.histogram(X, bins=ngrid, density=True)
plt.plot(bins[:-1], h, 'b.', linewidth=2, label='Sampled')
plt.plot(x[:-1], f[:-1], 'r', label='Theory')
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Normal distribution with μ = 0.2 and σ = 0.1')

# Bar plot of normalized frequency
plt.figure(4)
plt.bar(bins[:-1], h, width=deltax, alpha=0.6)
plt.plot(x[:-1], f[:-1], 'r', linewidth=2, label='Theory')
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Normal distribution with μ = 0.2 and σ = 0.1')

# Histogram object with normalization to PDF
plt.figure(5)
plt.hist(X, bins=ngrid, density=True, alpha=0.6, label='Sampled')
plt.plot(x[:-1], f[:-1], 'r', linewidth=2, label='Theory')
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Normal distribution with μ = 0.2 and σ = 0.1')

# Plot frequencies from histogram object
plt.figure(6)
plt.plot(bins[:-1], h, 'b.', label='Sampled')
plt.plot(x[:-1], f[:-1], 'r', label='Theory')
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Normal distribution with μ = 0.2 and σ = 0.1')

plt.show()