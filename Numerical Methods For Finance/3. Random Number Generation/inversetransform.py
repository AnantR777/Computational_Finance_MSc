import numpy as np
import matplotlib.pyplot as plt

# Plot and sample the distribution of X = cos U
ngrid = 200
nsample = 10000

# (a) Compute the PDF
dx = 1 / ngrid
x = np.linspace(0, 1, ngrid + 1)
f = 2 / (np.pi * np.sqrt(1 - x**2)) #Jarcobian Transformation - to recieve a different PDF.

# (b) Sample
dy = np.pi / ngrid
y = np.linspace(-np.pi / 2, np.pi / 2, ngrid + 1)
U = np.pi * (np.random.rand(nsample) - 0.5)
hu, _ = np.histogram(U, bins=y, density=True)
X = np.cos(U)
hx, _ = np.histogram(X, bins=x, density=True)

# Plot the PDF of U
plt.figure(1)
plt.plot(y[:-1], hu, 'r', label="Sampled")
plt.plot(y, np.ones_like(y) / np.pi, 'b', label="Theory")
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Distribution of U')

# Plot the PDF of X
plt.figure(2)
plt.plot(x[:-1], hx, 'r', label="Sampled")
plt.plot(x, f, 'b', label="Theory")
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.title('Distribution of X = cos U')
plt.show()