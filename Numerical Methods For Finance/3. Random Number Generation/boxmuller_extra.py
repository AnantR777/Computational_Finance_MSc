import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

#(1) generate U1 ∼ U[0, 1] and U2 ∼ U[0, 1].
#(2) θ := 2πU2, ρ := √−2 log U1
#(3) Z1 := ρ cos θ is ∼ N (0, 1)
#same as Z2 := ρ sin θ

# Define the parameters
mu = 0.2  # mean
sigma = 0.1  # standard deviation
a = -0.2  # left truncation
b = 0.6  # right truncation
nsteps = 200  # number of grid steps
npoints = 2000  # number of points

# Define the grid
deltax = (b - a) / nsteps  # grid step
x = np.linspace(a, b, nsteps)  # grid

# Box-Muller method for generating normally distributed random numbers

# The Box-Muller method is used to generate normally distributed random numbers
# from uniformly distributed random numbers. It is a widely used technique
# because of its simplicity and efficiency.

L = np.sqrt(-2 * np.log(np.random.rand(npoints//2)))
U = np.random.rand(npoints//2)
N1 = np.concatenate([L * np.cos(2 * np.pi * U), L * np.sin(2 * np.pi * U)])
N2 = mu + sigma * N1

# Inverse Box-Muller method
# The inverse Box-Muller method transforms normally distributed random variables
# back to uniformly distributed variables. This method is useful in probabilistic
# analyses and simulations where you need to understand the behavior of the inverse
# transformation.
R = N1**2 + N2**2
Theta = np.arctan2(N2, N1)

U1 = np.exp(-R/2)
U2 = Theta / (2 * np.pi)

# Jacobian of the inverse transformation
# The Jacobian is important in transformations as it represents the factor by which
# the transformation expands or shrinks volumes. In the context of probability distributions,
# it's crucial for understanding how densities transform under the change of variables.
Jacobian = np.abs(-0.5 * U1 * 1 / (2 * np.pi))

# Calculate the product of the PDFs for the generated random variables
fN1 = norm.pdf(N1, 0, 1)  # PDF of N1, normally distributed with mean 0 and std 1
fN2 = norm.pdf(N2, 0, 1)  # PDF of N2, normally distributed with mean 0 and std 1
product_of_PDFs = fN1 * fN2

# Calculate the product of the PDFs for the generated random variables
fN1 = norm.pdf(N1, 0, 1)  # PDF of N1, normally distributed with mean 0 and std 1
fN2 = norm.pdf(N2, 0, 1)  # PDF of N2, normally distributed with mean 0 and std 1
product_of_PDFs = fN1 * fN2

# Scatter plot for N1
# This scatter plot visualizes the generated random numbers (N1) against their probability
# density evaluated at those points. It's a way to visually inspect the distribution of the generated numbers.
U = np.random.rand(npoints)
plt.scatter(N1, U * norm.pdf(N1, mu, sigma), marker='.')
plt.plot(x, norm.pdf(x, mu, sigma))
plt.xlabel('x')
plt.ylabel('f_x')
plt.xlim([a, b])
plt.show()

# Scatter plot for N2
# Similar scatter plot for N2
U = np.random.rand(npoints)
plt.scatter(N2, U * norm.pdf(N2, mu, sigma), marker='.')
plt.plot(x, norm.pdf(x, mu, sigma))
plt.xlabel('x')
plt.ylabel('f_x')
plt.xlim([a, b])
plt.show()

# Output of Jacobian and product of PDFs
print("The absolute value of the Jacobian of the inverse transformation is ", Jacobian)
print("The product of the probability density functions fN1 and fN2 is ", product_of_PDFs)

# Histogram of Jacobian and product of PDFs
# This histogram shows the distribution of the values of the Jacobian and the product of the PDFs.
# It is useful for analyzing the characteristics of the transformation and the distribution of the generated numbers.
plt.figure()
plt.hist(Jacobian)
plt.hist(product_of_PDFs)
plt.xlabel('x')
plt.ylabel('f_x')
plt.xlim([a, b])
plt.show()