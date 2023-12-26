import numpy as np
import matplotlib.pyplot as plt

# Parameters for Laplace distribution
eta = 4
p = 0.5  # Probability skew parameter
nsample = 10**6  # Number of samples

# Grid parameters
a, b = -3, 3  # Truncation limits
deltax = 0.05
x = np.arange(a, b, deltax)  # Discretization of our grid
xx = x[:-1] + deltax / 2  # Shifted x-axis for bar chart

# PDF of Laplace Distribution
fX = p * eta * np.exp(-eta * x) * (x >= 0) + (1 - p) * eta * np.exp(eta * x) * (x < 0)

# Plotting the theoretical PDF
plt.figure(1)
plt.plot(x, fX, 'r')
plt.xlabel('x')
plt.ylabel('f_X')
plt.title('Laplace Distribution')

# Sampling from the Distribution
U = np.random.rand(nsample)
X = -1 / eta * np.log((1 - U) / p) * (U >= 1 - p) + 1 / eta * np.log(U / (1 - p)) * (U < 1 - p)

# Plotting the histogram of sampled data
plt.figure(2)
plt.hist(X, bins=x, density=True, alpha=0.6)
plt.plot(x, fX, 'r', linewidth=2)
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend(['Sampled', 'Theory'])
plt.title('Sampled Laplace Distribution')

# Corrected Bar Chart
plt.figure(3)

# Use np.histogram to get the counts
counts, _ = np.histogram(X, bins=x, density=True)

# Ensure the counts and xx have the same length
assert len(counts) == len(xx), "Counts and xx arrays must have the same length"

plt.bar(xx, counts, width=deltax, alpha=0.6)
plt.xlabel('x')
plt.ylabel('f_X')
plt.legend(['Sampled'])
plt.title('Laplace Distribution - Bar Chart')

plt.show()
