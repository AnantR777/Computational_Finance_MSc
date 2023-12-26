import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## Marsaglia Variant for Normally Distributed Random Deviates
# This is a modification of the Box-Muller method in order to avoid using
# trigonometric functions. We take advantage of their polar form.

# This method computes pairs of normally distributed random numbers.
# --------------------------------------------------------------------

# Parameters
a = -5 # left truncation
b = 5 # right truncation
deltax = 0.05
x = np.arange(a, b + deltax, deltax)  # Grid for the graphical checks

ndeviates = 10**6  # Number of deviates required


# Arrays for storing variables
W = np.ones(ndeviates)  # W stores squared magnitudes of vectors (V1, V2)
# Initially filled with ones to ensure the first entry into the while loop
# It checks if the point (V1[i], V2[i]) is inside the unit circle (W[i] < 1)

V1 = np.zeros(ndeviates)  # V1 stores the first coordinate of random points
V2 = np.zeros(ndeviates)  # V2 stores the second coordinate of random points
# These points are generated uniformly over the square [-1, 1] x [-1, 1]
# and then filtered to keep only those inside the unit circle


# Generate pairs of numbers until they fall inside the unit circle
for i in range(ndeviates):
    while W[i] >= 1:
        u1, u2 = np.random.random(2)
        V1[i], V2[i] = 2 * u1 - 1, 2 * u2 - 1  # Transforming to [-1, 1] range
        W[i] = V1[i]**2 + V2[i]**2  # Calculating the squared magnitude

# Apply the Marsaglia transformation
Z1 = V1 * np.sqrt(-2 * np.log(W) / W)
Z2 = V2 * np.sqrt(-2 * np.log(W) / W)



# Graphical check for Z1 - Histogram with theoretical overplot
plt.figure(1)
plt.hist(Z1, bins=np.arange(a, b, deltax), density=True, alpha=0.6)
plt.plot(x, norm.pdf(x, 0, 1), 'r', linewidth=2)
plt.xlim([a, b])
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend(['Sampled', 'Theory'])
plt.title('Standard Normal Distribution X ~ N(0,1)')

# Graphical check for Z2 - Histogram with overplot
plt.figure(2)
h2, bins, _ = plt.hist(Z2, bins=x, density=True, alpha=0.75)
plt.plot(x, norm.pdf(x, 0, 1), 'r', linewidth=2)
plt.xlim(a, b)
plt.xlabel('x')
plt.ylabel('f')
plt.legend(['Sampled', 'Theory'])
plt.title('Standard Normal Distribution X ~ N(0,1) for Z2')


def marsaglia_normal_deviate(n, mu=0, sigma=1):
    """
    Generates 'n' normally distributed random deviates with mean 'mu' and
    standard deviation 'sigma' using the Marsaglia method.

    Parameters:
    n : int
        The number of normally distributed deviates to generate.
    mu : float, optional
        The mean of the desired normal distribution. Default is 0.
    sigma : float, optional
        The standard deviation of the desired normal distribution. Default is 1.

    Returns:
    list
        A list of normally distributed random deviates with specified mean and standard deviation.
    """
    deviates = []
    while len(deviates) < n:
        u1, u2 = np.random.rand(), np.random.rand()
        v1, v2 = 2 * u1 - 1, 2 * u2 - 1
        w = v1 ** 2 + v2 ** 2

        if w < 1:
            z1 = v1 * np.sqrt(-2 * np.log(w) / w)
            z2 = v2 * np.sqrt(-2 * np.log(w) / w)
            # Transform the deviates to the desired mean and standard deviation
            deviates.append(mu + sigma * z1)
            if len(deviates) < n:
                deviates.append(mu + sigma * z2)

    return deviates


# Example usage with different mean and standard deviation
num_deviates = 100000
mu = 5  # Mean
sigma = 2  # Standard deviation
normal_deviates = marsaglia_normal_deviate(num_deviates, mu, sigma)

x = np.linspace(1, 9, 200)
plt.figure(3)
plt.hist(normal_deviates, bins=x, density=True)


## This section is to show that the (x1, x2) generated are distributed uniformly in S (−→ exercise) and serve as input for
# Box&Muller

def generate_transformed_pairs(n):
    """
    Generates 'n' pairs of transformed random numbers.

    Each pair consists of:
    - x1: the squared magnitude of a randomly generated vector within the unit circle.
    - x2: the angle of this vector in radians, normalized to the interval [0, 1].

    Parameters:
    n : int
        The number of pairs to generate.

    Returns:
    list, list
        Two lists containing the x1 and x2 values of the generated pairs.
    """

    def arg(x, y):
        """
        Calculates the normalized angle of the vector (x, y) in radians.

        Parameters:
        x : float
            The x-coordinate of the vector.
        y : float
            The y-coordinate of the vector.

        Returns:
        float
            The normalized angle of the vector, in the range [0, 1).
        """
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += 2 * np.pi
        return angle / (2 * np.pi)

    pairs_x1 = []
    pairs_x2 = []
    while len(pairs_x1) < n:
        u1, u2 = np.random.rand(), np.random.rand()
        v1, v2 = 2 * u1 - 1, 2 * u2 - 1
        w = v1 ** 2 + v2 ** 2
        if w < 1:
            x1 = w
            x2 = arg(v1, v2)
            pairs_x1.append(x1)
            pairs_x2.append(x2)

    return pairs_x1, pairs_x2


n = 1000
pair_x1, pair_x2 = generate_transformed_pairs(n)

plt.figure(4)
plt.scatter(pair_x1, pair_x2)
plt.xlabel('Squared Magnitude (x1)')
plt.ylabel('Normalized Angle (x2)')
plt.title('Transformed Random Pairs')
plt.show()