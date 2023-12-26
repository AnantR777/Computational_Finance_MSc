import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Define the parameters
mu = 2/3  # mean
nsample = 10**6  # number of random samples

# Define the grid
x = np.linspace(0,5,101)
# x = np.arange(0, 5.05, 0.05)  # doesn't include 5.05

# Compute the PDF and the CDF
# f = 1/mu*exp(-x/mu);
# F = 1-exp(-x/mu);
f = expon.pdf(x, scale=mu) # scale is the mean of the distribution
F = expon.cdf(x, scale=mu)

# Plot the PDF and the CDF
plt.figure(1) # after stating figure, state plots you want on the figure
plt.plot(x, f, 'r', label='PDF')
plt.plot(x, F, 'b', label='CDF')
plt.xlabel('x')
plt.legend()
plt.title('Exponential distribution with μ = 2/3')
plt.savefig('exponential.pdf')  # Save the figure to a file

## Sample an exponential distribution
U = np.random.rand(nsample)  # standard uniform random numbers
X = -mu * np.log(U)  # transformation to exponential random numbers

# Bin the random variables in a histogram and normalize it
dx = x[1] - x[0]  # bin width, they are all equal so just take diff of first 2
h, xx = np.histogram(X, bins=x, density=True) # bins are in 0, 0.05, ..., 5
# h is the frequency/prob density
# outputs (frequency density, bin edges)

# Plot sampled vs theoretical
plt.figure(2)
# bin centres are x values plus half of width
plt.plot(xx[:-1] + dx/2, h, 'b', label='Sampled') # xx[:-1] + dx/2 shifts all but last bin edges to right by half bin width
plt.plot(x, f, 'r', label='Theory')
plt.xlim([0, 5])
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Exponential distribution with μ = 0.2')

# Bar plot
plt.figure(3)
plt.bar(xx[:-1] + dx/2, h, width=dx) # width of bar is just bin width, note there are 500 centres, 501 edges - we exclude the last edge
plt.plot(x, f, 'r', label='Theory')
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Exponential distribution with μ = 0.2')
plt.show()