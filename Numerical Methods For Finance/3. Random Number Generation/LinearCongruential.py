import numpy as np
import matplotlib.pyplot as plt

##  Linear Congruential Generator for Uniform Random Deviates

# Parameters
a = 1597
b = 51749
M = 244944
N0 = 40243 # % Seed number

ndeviates = 1000 # Number of deviates required

## Linear Congruential Generator for Uniform Random Deviates


# Initialise zeros matrix for number of deviates
A = np.zeros(ndeviates)
# Initialize the first element of A
A[0] = (a * N0 + b) % M

# Generate pseudorandom numbers using LCG
for i in range(1, ndeviates):
    A[i] = (a * A[i - 1] + b) % M

# Transform numbers into standard uniform deviates
U = A / M

# Plot the figure to see how they lie
# They could fall onto parallel lines:
# e.g. When a = 1229; b=1; M = 2048
# Or they could fall (relatively) well distributed:
# e.g. When a = 1597; b = 51749; M = 244944

nbins = 100

# Probability density function
plt.figure(1)
plt.hist(U, bins=nbins, density=True, alpha=0.7, label="Sample")
plt.plot(np.linspace(0, 1, nbins+1), np.ones(nbins+1), label="Theory")
plt.xlim(-0.2, 1.2)
plt.ylim(0, 2)
plt.xlabel('x')
plt.ylabel('f_U')
plt.legend()
plt.title('Probability density function')

# 2D scatter plot
plt.figure(2)
plt.plot(U[:-1], U[1:], '.')
plt.xlabel('U_i')
plt.ylabel('U_{i+1}')
plt.title('2D scatter plot')

# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[:-2], U[1:-1], U[2:], marker='.')
ax.set_xlabel('U_i')
ax.set_ylabel('U_{i+1}')
ax.set_zlabel('U_{i+2}')
plt.title('3D scatter plot')

## Method Using Function

def linear_cong_gen(N_0, a, b, M, n):
    """
    Generates a sequence of pseudorandom numbers using the linear congruential generator (LCG) method and visualizes the distribution.

    Parameters:
    N_0 : int
        The initial seed or starting value.
    a : int
        The multiplier in the LCG formula.
    b : int
        The increment in the LCG formula.
    M : int
        The modulus in the LCG formula.
    n : int
        The number of pseudorandom numbers to generate.

    Returns:
    None
        Displays a histogram and a scatter plot of the generated numbers.
    """
    # Initialize the array for the sequence
    N = np.zeros(n)
    N[0] = N_0 % M  # Initial seed value

    # Generate the sequence
    for i in range(1, n):
        N[i] = (a * N[i - 1] + b) % M  # Linear congruential formula

    # Normalize the sequence to [0, 1) interval
    U = N / M

    # Plotting the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram of the normalized values
    ax[0].hist(U, density=True)
    ax[0].set_xlabel('Value')
    ax[0].set_title('Empirical PDF Histogram')
    ax[0].set_ylabel('Density')

    # Scatter plot of successive values to check for independence
    ax[1].scatter(U[:-1], U[1:], alpha=0.2, color='b')
    ax[1].set_title('Scatter Plot in the (Ui-1, Ui) Plane')
    ax[1].set_xlabel('U[i-1]')
    ax[1].set_ylabel('U[i]')

    plt.show()

# Example usage of the function
linear_cong_gen(N_0=1000, M=2048, a=1229, b=1, n=10000)
