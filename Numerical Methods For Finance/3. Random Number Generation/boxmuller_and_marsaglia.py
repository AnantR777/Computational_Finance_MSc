import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

#(1) generate U1 ∼ U[0, 1] and U2 ∼ U[0, 1].
#(2) θ := 2πU2, ρ := √−2 log U1
#(3) Z1 := ρ cos θ is ∼ N (0, 1)
#same as Z2 := ρ sin θ

# Define the parameters
a = -4  # left truncation
b = 4  # right truncation
ngrid = 200  # number of grid steps
nsample = 2000  # number of random samples

# Define the grid
deltax = (b-a)/ngrid  # grid step
x = np.linspace(a, b, ngrid)  # grid

# Box-Muller
U1 = np.random.rand(nsample//2)
U2 = np.random.rand(nsample//2)
start_time_bm = time.perf_counter()
rho = np.sqrt(-2*np.log(U1))
theta = 2*np.pi*U2
N = np.concatenate((rho * np.cos(theta), rho * np.sin(theta)))  # standard normal numbers
# ensure the two sets of numbers you generate are independent of one another
end_time_bm = time.perf_counter()
box_muller_time = end_time_bm - start_time_bm

# Scatter plot for Box-Muller
U = np.random.rand(nsample)
plt.figure()
plt.scatter(N, U * norm.pdf(N), alpha=0.5)
# randomly scatter the points along the y-axis from 0 up to the value of the PDF at each corresponding N-value.
# This creates a visualization where the density of points reflects the PDF of the distribution
# The more points there are in a vertical slice, the higher the probability density for that value of N
plt.plot(x, norm.pdf(x), 'r', linewidth=2)
plt.xlabel('x')
plt.ylabel('f_x')
plt.title('Box-Muller Algorithm')

# Marsaglia polar method (variant of Box-Muller)
start_time_marsaglia = time.perf_counter()
V1 = 2 * U1 - 1
V2 = 2 * U2 - 1
W = V1**2 + V2**2
mask = W <= 1 # list of booleans such that W values are less than 1
WI = W[mask]
rho2 = np.sqrt(-2 * np.log(WI) / WI)
N1 = np.concatenate((rho2 * V1[mask], rho2 * V2[mask]))  # standard normal numbers
# note V1[mask]/sqrt(WI) and V2[mask]/sqrt(WI) are cos and sin of the angle (x,y) makes with x-axis
end_time_marsaglia = time.perf_counter()
marsaglia_time = end_time_marsaglia - start_time_marsaglia
L = len(N1)
L_nsample_ratio = L / nsample # ratio of accepted to all samples

# Scatter plot for Marsaglia
plt.figure()
plt.scatter(N1, U[:L] * norm.pdf(N1), alpha=0.5)
# randomly scatter the points along the y-axis from 0 up to the value of the PDF at each corresponding N1-value.
# This creates a visualization where the density of points reflects the PDF of the distribution
# The more points there are in a vertical slice, the higher the probability density for that value of N1
plt.plot(x, norm.pdf(x), 'r', linewidth=2)
plt.xlabel('x')
plt.ylabel('f_x')
plt.title('Marsaglia Polar Method')

# Display the time taken by both methods
print("Box-Muller Time: ", '{0:.16f}'.format(box_muller_time))
print("Marsaglia Time: ", '{0:.16f}'.format(marsaglia_time))
print("Ratio of Marsaglia samples to nsample: ", L_nsample_ratio)
# note marsgalia takes less time
plt.show()