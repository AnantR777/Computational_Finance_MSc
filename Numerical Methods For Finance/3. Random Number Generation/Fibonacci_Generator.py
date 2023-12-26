import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# STEP 1: Congruential Random Generator
# ----------------------------------
# This is required for the first X numbers, before moving on to the
# Fibonacci generator for the remaining

# Parameters
a = 1597
b = 51749
M = 2**30
N0 = 25 # Seed number
lcgnum = 17 #Number of linear congruential unif rand deviants required

ndeviates = 10000 # Number of deviates required

# Initialize A as an array of zeros
A = np.zeros(ndeviates)

# Initialize the first element of A
A[0] = (a * N0 + b) % M

# Generate pseudorandom numbers using LCG
for i in range(ndeviates - 1):
    A[i + 1] = (a * A[i] + b) % M

U = A / M

# ----------------------------------
# STEP 2: Fibonacci Random Generator
# ----------------------------------

# Assuming you have the array U and ndeviates already defined

# Initialize arrays X and Y
X = np.zeros(ndeviates - 1)
Y = np.zeros(ndeviates - 1)

# Generate pseudorandom numbers using Fibonacci Random Generator
for i in range(lcgnum, ndeviates):
    U_i = U[i - 17] - U[i - 5]
    if U_i < 0:
        U_i += 1
    U = np.append(U, U_i)  # Append the new U_i to the U array

# Create arrays X and Y for plotting
X = U[:ndeviates - 1]
Y = U[1:ndeviates]

# Plot in the [U(i), U(i+1)] plane
plt.figure(1)
plt.plot(X, Y, 'r.')

## Using Function


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

    return U



def fibonacci_gen(seed, mu, vu, n):
    """
    Generates a sequence of pseudorandom numbers using a Fibonacci-like generator
    and visualizes the distribution.

    Parameters:
    seed : array-like
        Initial seed values, a list or array of numbers to start the generation.
    mu : int
        The first lag parameter.
    vu : int
        The second lag parameter, usually larger than 'mu'.
    n : int
        The number of pseudorandom numbers to generate.

    Returns:
    None
        Displays a histogram and a scatter plot of the generated numbers.
    """
    seed = list(seed)  # Ensure the seed is mutable
    i = mu
    j = vu - 1

    while len(seed) < n:
        i = vu if i == 0 else i - 1
        j = vu if j == 0 else j - 1
        zi = seed[j] - seed[i]

        if zi <0:
              zi += 1
        seed = np.insert(seed, i, zi)


    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram of the values
    ax[0].hist(seed, density=True)
    ax[0].set_xlabel('Value')
    ax[0].set_title('Empirical PDF Histogram')
    ax[0].set_ylabel('Density')

    # Scatter plot of successive values
    ax[1].scatter(seed[:-1], seed[1:], alpha=0.2, color='b')
    ax[1].set_title('Scatter Plot in the (Ui−1, Ui) Plane')
    ax[1].set_xlabel('U[i-1]')
    ax[1].set_ylabel('U[i]')

    plt.show()

# Example usage
seed = linear_cong_gen(N_0=10, a=1366, b=150889, M=714025, n=17)
fibonacci_gen(seed=seed, vu=17, mu=5, n=5000)
