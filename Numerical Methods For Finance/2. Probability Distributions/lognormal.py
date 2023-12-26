import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Define the parameters
mu = 0.2
sigma = 0.1
nsample = 10**6

# Define the grid
# x = np.linspace(0,2,101)
x = np.arange(0, 2.02, 0.02)  # doesn't include 2.02

# Compute the PDF and CDF using the lognorm functions from scipy.stats
s = sigma  # The shape parameter for lognorm is the sigma in the lognormal distribution
f = lognorm.pdf(x, s, scale=np.exp(mu))  # The scale parameter is the median e^mu
F = lognorm.cdf(x, s, scale=np.exp(mu))

# Plot the PDF and CDF
plt.figure(1)
plt.plot(x, f, 'r', label='PDF')  # Red line for the PDF
plt.plot(x, F, 'b', label='CDF')  # Blue line for the CDF
plt.xlabel('x')
plt.legend()
plt.title('Lognormal distribution with μ = 0.2 and σ = 0.1')
plt.savefig('lognormal.pdf')  # Save the figure

# Sample a normal distribution and then sample the lognormal distribution
# by the exponential transformation
# U = np.random.rand(nsample)  # standard uniform random numbers
# Convert uniform random numbers to normal distribution (inverse transform method)
# X = mu + sigma * norm.ppf(U) # method 1a
# X = norm.ppf(U, loc=mu, scale=sigma) # method 1b
# X = mu + sigma * np.random.randn(nsample) # method 2 without using inverse cdf
X = np.random.normal(mu, sigma, nsample) # method 3, getting normal directly
Y = np.exp(X)

# Plot the histogram with lognormal fit
plt.figure(2)
plt.hist(Y, 100, density=True, alpha=0.75, label='Sampled') # alpha is transparency for plot
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = lognorm.pdf(x, s, scale=np.exp(mu)) # same length as x
plt.plot(x, p, 'k', linewidth=2, label='Lognormal fit')
plt.legend()
plt.title('Lognormal random variables with μ = 0.2 and σ = 0.1')

# Bin the random variables in a histogram
plt.figure(3)
count, bins, ignored = plt.hist(Y, bins=x, density=True, color='blue', alpha=0.7) # returns frequency and centres
plt.plot(bins[:-1] + (bins[1]-bins[0])/2, count) # plot of frequency vs bin centres
plt.title('Lognormal random variables with μ = 0.2 and σ = 0.1')

# Bar plot of the binned random variables
plt.figure(4)
plt.bar(bins[:-1], count, width=bins[1]-bins[0], color='blue', alpha=0.7) # use all but last bin edges as bar centres
plt.title('Lognormal random variables with μ = 0.2 and σ = 0.1')

plt.show()