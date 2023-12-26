import numpy as np

# Define a random mean vector (mu) and a covariance matrix (Sigma)
# For demonstration, let's consider a 2-dimensional case
mu = np.array([0.5, -0.3])  # mean vector
Sigma = np.array([[1.0, 0.6], [0.6, 1.0]])  # covariance matrix

# Step 1: Cholesky Decomposition of Sigma
C = np.linalg.cholesky(Sigma)

# Step 2: Generate Z ~ N(0, I)
n = len(mu)  # Dimension of the normal
Z = np.random.randn(n)  # Standard normal random variables

# Step 3: Transform to get correlated normal deviates
correlated_normal_deviates = mu + C.dot(Z)

print(correlated_normal_deviates)
# note we only generate 2 correlated here

def marsaglia_method(n):
    pair = []
    while len(pair) < n:
        u1, u2 = np.random.rand(), np.random.rand()
        v1 = 2 * u1 - 1
        v2 = 2 * u2 - 1
        w = v1**2 + v2**2
        if w < 1:
            z1 = v1 * np.sqrt(-2 * np.log(w) / w)
            z2 = v2 * np.sqrt(-2 * np.log(w) / w)
            pair.extend([z1, z2])
    return pair[:n]

# Define a random mean vector (mu) and a covariance matrix (Sigma) for 4 dimensions
mu = np.array([0.5, -0.3, 0.1, -0.2])  # mean vector
Sigma = np.array([[1.0, 0.6, 0.2, 0.1],
                  [0.6, 1.0, 0.3, 0.2],
                  [0.2, 0.3, 1.0, 0.5],
                  [0.1, 0.2, 0.5, 1.0]])  # covariance matrix

# Step 1: Cholesky Decomposition of Sigma
C = np.linalg.cholesky(Sigma)

# Step 2: Generate Z ~ N(0, I) using Marsaglia's method
n = len(mu)  # Dimension of the normal
Z = np.array(marsaglia_method(n))  # Standard normal random variables using Marsaglia's method

# Step 3: Transform to get correlated normal deviates
correlated_normal_deviates = mu + C.dot(Z)

print(correlated_normal_deviates)
#note we generate 4 correlated here