import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
import scipy.stats as stats

# takes time to run

## Example: Normally Dist Random Deviates from Laplace Distribution

# We will seek to find normally distributed random deviates by using the
# Laplace distribution as the majorant function.
#  Recall Laplace Distribution PDF is
#  (1/2b)*exp( -abs(x-mu)/b )
# Since we are looking for standard normally distributed deviates we can
# take:
# b = 1 ...AND... mu = 0

# Number of random deviates required
nsample = 10**6

# Constant for multiplying the majorant function
# This is calculated analytically based on the relationship between the target
# and majorant distributions.
c = np.sqrt(2 * np.exp(1) / np.pi)

# Generate uniformly distributed random numbers
U = np.random.rand(nsample)

# Generate Laplace distributed random numbers
# The Laplace distribution is used as the majorant distribution.
X = -1 * np.log(2 * (1 - U)) * (U >= 0.5) + 1 * np.log(2 * U) * (U < 0.5)

# Majorant Function:
# Calculate the PDF of the Laplace distribution for the generated random numbers.
g = 0.5 * np.exp(-1 * np.abs(X))

# Minor Function:
# Calculate the PDF of the Normal distribution for the generated random numbers.
f = stats.norm.pdf(X)

# Scale our Majorant function as required
maj = c * np.random.rand(nsample) * g

# Acceptance-rejection algorithm:
# Test whether each scaled majorant value is less than or equal to the corresponding minor value.
N = np.where(maj <= f, X, 0)

# Gather the non-zero elements
# These are the accepted values that follow the target distribution.
Z = N[N != 0]

# Z now contains normally distributed random deviates generated using the acceptance-rejection method.

## Stats and plots
# Lower and upper truncation limits
a, b = -4, 4

# Grid step and grid for plotting
deltax = 0.1
x = np.arange(a, b + deltax, deltax)

# Acceptance ratio
accept_ratio = len(Z) / nsample
print("Acceptance Ratio:", accept_ratio)

# Output the value of c
analytical_c = c
print("Analytical c:", analytical_c)

# Calculate numerical c (for comparison)
numerical_c = max(stats.norm.pdf(x) / (0.5 * np.exp(-np.abs(x))))
print("Numerical c:", numerical_c)

# Plotting
plt.figure(figsize=(10, 6))
plt.hist(Z, bins=x, density=True, alpha=0.4, color='blue', label="Sampled f(x)")
fx = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
gx = 0.5 * np.exp(-np.abs(x))
plt.plot(x, fx, label='Theoretical f(x)')
plt.plot(x, c * gx, label='Majorant function c*g(x)', color='green')
plt.xlabel('x')
plt.legend()
plt.title('Standard Normal Distribution using the Accept-Reject Algorithm')

# Create a new figure for plotting
plt.figure(2)

# Plot the quadratic function x^2 - 2x + 1
# This function is not directly related to the acceptance-rejection method but might be used
# for illustrative purposes or to show specific properties of the function.
plt.plot(x, x**2 - 2 * x + 1, label="x^2-2x+1")

# Plot the ratio of the target distribution (f) and the proposal distribution (g)
# This ratio is crucial in the acceptance-rejection method as it helps determine
# the scaling factor c that ensures the proposal distribution envelopes the target distribution.
plt.plot(x, fx / gx, label="f/g")

# Plot a horizontal line at the value c = sqrt(2e/pi)
# This value of c is the scaling factor used in the acceptance-rejection method.
# It represents the maximum value of the f/g ratio, ensuring that c*g(x) is always
# greater than or equal to f(x) for all x, which is a prerequisite for the acceptance-rejection method.
plt.plot(x, c * np.ones_like(x), '--g', label='c = (2e/pi)^{1/2}')

# Set the x-axis limits to focus on the range of interest (0 to 3).
plt.xlim(0, 3)

# Label the x-axis
plt.xlabel('x')

# Add a legend to the plot to identify each curve
plt.legend()

# Set the title of the plot
plt.title('x^2-2x+1 = 0 where f/g = max = c')


def target_distribution(x):
    """
    Calculate the probability density of the target distribution at the given points.
    Here, the target distribution is the standard normal distribution.

    Parameters:
    x : float or array-like
        Points at which to evaluate the PDF.

    Returns:
    float or array-like
        The PDF values of the target distribution at points 'x'.
    """
    return norm.pdf(x, loc=0, scale=1)


def proposal_distribution():
    """
    Generate a single sample from the proposal distribution.
    Here, the proposal distribution is the Laplace distribution.

    Returns:
    float
        A random sample from the Laplace distribution.
    """
    return np.random.laplace(loc=0, scale=1)


def acceptance_rejection(n):
    """
    Generate samples from the target distribution using the acceptance-rejection method.

    Parameters:
    n : int
        Number of samples to generate.

    Returns:
    list
        Samples from the target distribution.
    """
    samples = []
    c = 1.6  # Scaling factor to ensure the proposal distribution envelopes the target distribution

    while len(samples) < n:
        x_proposal = proposal_distribution()
        u = np.random.uniform(0, 1)
        # Acceptance criterion
        if u * c * laplace.pdf(x_proposal) <= target_distribution(x_proposal):
            samples.append(x_proposal)

    return samples


def plot_results(samples):
    """
    Plot the results of the sampling process.

    Parameters:
    samples : list
        Samples generated by the acceptance-rejection method.
    """
    plt.figure(3)
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='g',
             label='Accepted Samples')
    x = np.linspace(-5, 5, 1000)

    plt.plot(x, target_distribution(x), 'r-', lw=2, label='Target Distribution')
    plt.legend()
    plt.show()


# Generate samples and plot results
num_samples = 100000
samples = acceptance_rejection(num_samples)
plot_results(samples)

