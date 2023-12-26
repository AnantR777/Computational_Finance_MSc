import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
from scipy.special import iv

# Define the parameters
d = 5  # degrees of freedom
lambda_ = 2  # non-centrality parameter, sum of (mean of N_i)^2
a = 0  # left truncation
b = 20  # right truncation
ngrid = 200  # number of grid intervals
nsample = 10**6  # number of random samples

# Define the grid
x = np.linspace(a, b, ngrid+1)
deltax = x[1] - x[0]  # grid step

# Compute and plot the PDF and CDF
f = ncx2.pdf(x, d, lambda_) # specify degrees of freedom and non-centrality parameter
F = ncx2.cdf(x, d, lambda_)
bessel_form = 1/2 * np.exp(-(x+lambda_)/2) * (x/lambda_)**(d/4 - 1/2) * iv(d / 2 - 1, np.sqrt(lambda_ * x))

# Plot the PDF and CDF
plt.figure(1)
plt.plot(x, f, 'r', label='PDF', linewidth=2)
plt.plot(x, F, 'b', label='CDF', linewidth=2)
plt.plot(x, bessel_form, linestyle = '--', lw=1, alpha=0.8, label='Bessel Form pdf')
plt.xlabel('x')
plt.legend()
plt.title('Non-central chi-square PDF and CDF with d=5 and λ=2')
plt.savefig('ncchisq.png')

# Generate random samples from the non-central chi-square distribution using the inverse CDF
# U = np.random.rand(nsample)
# X = ncx2.ppf(U, d, lambda_)
# Sample the non-central chi-square distribution
X = ncx2.rvs(d, lambda_, size=nsample)

# Bar histogram
plt.figure(2)
bins = np.linspace(a, b, ngrid+1)
plt.hist(X, bins=bins, density=True, alpha=0.6) # note bins can be either number of bins or sequence of edges
plt.plot(x, f, 'r', linewidth=2)
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend(['Sampled', 'Theory'])
plt.title('Non-central chi-square PDF with d=5 and λ=2')

# Bar histogram with shifted bins
samples = np.random.noncentral_chisquare(d, lambda_, size=nsample)
x2 = np.concatenate([x - deltax / 2, [x[-1] + deltax / 2]])

plt.figure(3)
plt.hist(samples, bins = x2, density = True);
plt.plot(x, ncx2.pdf(x, d, lambda_), lw=1, c = 'r', alpha=0.8, label='ncx2 pdf', linestyle = '-.',)
plt.xlabel("x")
plt.ylabel("f_x")
plt.title('Non-central chi-square PDF with d=5 and λ=2, shifted bins')

# Line histogram
plt.figure(4)
counts, _ = np.histogram(X, bins=bins, density=True) # returns frequency
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.plot(bin_centers, counts, 'b', linewidth=2)
plt.plot(x, f, 'r--', linewidth=2)
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend(['Sampled', 'Theory'])
plt.title('Non-central chi-square PDF with d=5 and λ=2')

# Scatter plot
plt.figure(5)
U = np.random.rand(1000)
plt.scatter(X[:1000], U * ncx2.pdf(X[:1000], d, lambda_), alpha=0.6)
# randomly scatter the points along the y-axis from 0 up to the value of the PDF at each corresponding x-value.
# This creates a visualization where the density of points reflects the PDF of the distribution
# The more points there are in a vertical slice, the higher the probability density for that value of x
plt.plot(x, ncx2.pdf(x, d, lambda_), 'r', linewidth=2) # all samples lie under this curve
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend(['Sampled', 'Theory'])
plt.title('Non-central chi-square PDF with d=5 and λ=2')

plt.show()