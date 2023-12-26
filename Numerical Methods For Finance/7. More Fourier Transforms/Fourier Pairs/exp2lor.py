import numpy as np
import matplotlib.pyplot as plt

# takes a minute to run

# =========================
#  Laplace <--> Lorentzian
# =========================

# This script demonstrates the Fourier transform relationship between a bilateral
# exponential function in the time domain and a Lorentzian function in the frequency domain.
# The bilateral exponential, often used in signal processing, transforms into a Lorentzian
# distribution, which is a fundamental function in physics and mathematics.

# To help us understand how the FFT algorithm works, we can perform the
# fourier transform on functions that we already have an analytical
# solution to and then compare.

## STEP 1: Grids

# Since we are working with a numerical algorithm we need an appropriate
# grid over which to work. As a rule its always best to define the number
# of grid points to be a power of 2.

# Essentially there are two grids we need to consider:
#   - Real space
#   - Fourier space

# These two grids are related through the Nyquist relation. But which
# version of the relation depends on which space we are working in:
#   1. Pulse Space
#   dx * dxi = 2*pi/N
#
#   2. Frequency Space
#   dx * dnu = 1/N

# The difference lies in the factor 2*pi and we need to be careful to
# understand which one is correct.

# GRID IN REAL SPACE
N = 2048  # Number of grid points - determines the resolution in both real and Fourier spaces
dx = 0.01 # Grid step in real space - spacing between individual points
x = dx * np.arange(-N / 2, N / 2)  # Grid in x real  space
upperx = N * dx  # Upper truncation limit in real space

# GRID IN FOURIER SPACE (Pulsation)
dxi = (2*np.pi)/(N*dx) # Grid step size in fourier space / Spacing of the xi grid; Nyquist relation: Dx*Dxi = 2*pi/N
upperxi = N*dxi # Upper truncation limit in fourier space ; W = 2*pi/dx
xi = dxi * np.arange(-N / 2, N / 2) # Grid in fourier space


# GRID IN FOURIER SPACE (Frequency)
dnu = 1/(N*dx) # Grid step size in fourier space
uppernu = N*dnu # Upper truncation limit in fourier space
nu = dnu * np.arange(-N / 2, N / 2) # Grid in fourier space

# Notice that we use shift the grid points to be centred at 0 and symmetric
# either side. We will need to apply a correction to account for this later


## STEP 2: Analytical expressions & Functions

# Laplace function:
# f(x) = 1/2b * exp(-abs(x)/b), with b: scale parameter; or
# f(x) = a/2 * exp(-a*abs(x)),  with a: activity parameter
# We can see that a (or lambda) = 1/b

# Lorentz function (Pulsation):
# f(xi) = 1 / (1 + b^2 xi^2); or
# f(xi) = a^2 / (a^2 + xi^2)

# Lorentz function (Frequency):
# f(nu) = 1 / (1 - (2*b*pi*nu)^2); or
# f(nu) = a^2 / (a^2 + (2*pi*nu)^2)

# We will work with the second form as its easier for inputs

# ANALYTICAL expressions
# -----------------------------
a = 1 # Activity parameter

fa = 0.5*a*np.exp(-a*abs(x)) # Laplace
# We will use this to check that the inverse numerical FFT does a good
# approximation of the analytical expression


Fa_p = a**2/(a**2 + xi**2) #Lorentz (Pulsation)
# We will use this to check that the numerical FFT does a good
# approximation of the analytical expression

Fa_f = a**2/(a**2 + (2*np.pi*nu)**2) # Lorentz (Frequency)
# We will use this to check that the numerical FFT does a good
# approximation of the analytical expression

## STEP 3: Numerical Approximation

# NUMERICAL approximations
# -----------------------------
# Unfortunately, the definition of the FFT algorithm uses a different
# format for the Fourier Transform. Where we are used to the the FT having
# a positive exponent and FT^-1 having a negative component, the algorithm
# does the opposite, meaning we need to use:
#   ifft for our Fourier Transform
#    fft for our inverse Fourier Transform

# The algorithm was also designed such that the point 0 is the further left
# grid point, i.e. index(1). But our grid is symmetrically defined over the
# interval [-N/2:N/2], so we need to use the i/fftshift function.
# This works by swapping the positions of the vector to put the zero
# position in the 'correct' place. E.g.
# [-3,-2,-1,0,1,2] ---> [0,1,2,-3,-2,-1]
# But we need to undo this correction afterwards

Fn = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(fa))) * upperx
# Fn is the numerical Fourier transform of the function 'fa'. It is calculated
# using FFT. The fftshift operation is used to shift the zero-frequency
# component to the center of the spectrum.

fn_p = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Fa_p))) / upperx
fn_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Fa_f))) / upperx
# fn is the numerical inverse Fourier transform of 'Fa_p'. This operation is
# intended to transform the frequency domain representation back to the
# time domain (or original function space).


# Computing the Fourier transform and its inverse using different normalization factors
Fn1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(fa))) * dx
# Fn1 is similar to 'Fn' but scaled by dx instead of upper limit. This affects the
# amplitude of the transformed function, demonstrating the impact of scaling
# in the Fourier transform.

fn1_p = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Fa_p))) / dx
fn1_f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Fa_f))) / dx
# fn1 is the scaled inverse Fourier transform of 'Fa_p', similar to 'fn' but
# scaled by dx instead of upper limit. This represents the inverse operation to Fn1, showing how
# the function transforms back with the scaling factor.

## ALT: DTF Version

###############################################
# Freq version

# Initialize an array of zeros to store the DFT results. It's the same size as the input signal and capable of storing complex numbers.
# The 'dtype=complex' ensures that the array can store complex numbers, as DFT results are generally complex.
Fn2_f = np.zeros(N, dtype=complex)

# This is a nested loop to compute the DFT. The outer loop iterates over each frequency bin 'j'.
for j in range(N):  # Iterating over each frequency bin 'j'.
    # The inner loop iterates over each time point 'k' of the signal.
    for k in range(N):  # Iterating over each time point 'k' of the signal.
        # Compute the DFT using the formula:
        # Fn2[j] += exp(i * 2 * pi * j * k / N) * fa[k] * dx
        # The complex exponential function is a key component of the Fourier transform.
        # 'j / N' and 'k' serve as normalized frequency and time indices, respectively.
        Fn2_f[j] += np.exp(1j * 2 * np.pi * j * k / N) * fa[k] * dx


# Initialize an array of zeros for the IDFT results, similar to the DFT.
fn2_f = np.zeros(N, dtype=complex)

# This is a nested loop to compute the IDFT. The outer loop iterates over each time point 'n'.
for n in range(N):  # Iterating over each time point 'n'.
    # The inner loop iterates over each frequency bin 'k'.
    for k in range(N):  # Iterating over each frequency bin 'k'.
        # The IDFT formula:
        # fn2_f[n] += Fn2_f[k] * exp(-i * 2 * pi * k * n / N)
        # Here, Fn2_f[k] is the frequency domain data from the DFT.
        # The complex exponential has the opposite sign in the exponent compared to DFT.
        fn2_f[n] += Fn2_f[k] * np.exp(-1j * 2 * np.pi * k * n / N)

# Normalize the IDFT result by dividing by the number of points 'N'.
# This normalization is required as the sum of the complex exponentials scales with the number of points 'N'.
fn2_f /= N

#######################################
# Pulsation version
# Initialize an array of zeros to store the DFT results for the pulsation version.
# The array is of the same size as the input signal and is capable of storing complex numbers.
# The 'dtype=complex' ensures that the array can handle complex numbers, as DFT results are generally complex.
Fn2_p = np.zeros(N, dtype=complex)

# Nested loop to compute the DFT for the pulsation version.
for j in range(N):  # Iterating over each frequency bin 'j'.
    for k in range(N):  # Iterating over each time point 'k' of the signal.
        # Compute the DFT using the formula for the pulsation version:
        # Fn2_p[j] += exp(i * xi[j] * x[k]) * fa[k] * dx
        # This formula applies the complex exponential function, a key component of the Fourier transform.
        # 'xi[j]' is the angular frequency for the j-th frequency bin,
        # 'x[k]' is the time value for the k-th time point,
        # 'fa[k]' is the signal value at time 'x[k]',
        # and 'dx' is the time step size.
        # This calculation sums the contributions of the time-domain signal 'fa[k]' to the frequency-domain representation at 'xi[j]'.
        Fn2_p[j] += np.exp(1j * xi[j] * x[k]) * fa[k] * dx


# Initialize an array of zeros for the IDFT results in the pulsation version, similar to the DFT.
fn2_p = np.zeros(N, dtype=complex)

# Nested loop to compute the IDFT for the pulsation version.
for n in range(N):  # Iterating over each time point 'n'.
    for k in range(N):  # Iterating over each frequency bin 'k'.
        # Compute the IDFT using the formula for the pulsation version:
        # fn2_p[n] += Fn2_p[k] * exp(-i * xi[k] * x[n])
        # Here, 'Fn2_p[k]' is the frequency domain data (result from DFT) for the pulsation version.
        # The complex exponential function is used with a negative sign in the exponent,
        # which is the opposite of what is used in DFT.
        # 'xi[k]' is the angular frequency for the k-th frequency bin,
        # and 'x[n]' is the time value for the n-th time point.
        # This calculation reconstructs the time-domain signal from its frequency-domain representation.
        fn2_p[n] += Fn2_p[k] * np.exp(-1j * xi[k] * x[n])

# Normalize the IDFT result by dividing by the total time period.
# This normalization accounts for the scaling effect due to the sum of complex exponentials over the time period.
fn2_p /= N * dx

## STEP 4: Plotting

# Graphical check

# Plot 1: Inverse Fourier Transform in Pulsation Space
# This plot is used to compare the numerically inverted Fourier transform (in pulsation space) with the original Gaussian function.
plt.figure(1)
plt.plot(x, fn_p.real, 'ro', label='Re(fn)')  # Plot the real part of the inverse Fourier transform (fn_p)
plt.plot(x, fn_p.imag, 'g', label='Im(fn)')  # Plot the imaginary part of the inverse Fourier transform (fn_p)
plt.plot(x, fa, '--k', label='fa')  # Plot the original Gaussian function for comparison
plt.title('Pulsation Space: Numerical Inversion of Analytical FT')  # Title indicating this is the inverse FT in pulsation space
plt.xlabel('x')  # Label for the x-axis
plt.ylabel('f')  # Label for the y-axis
plt.legend()  # Show legend to identify the plots
plt.axis([-10, 10, 0, 0.6])  # Set the axis limits for better visualization

# Plot 2: Inverse Fourier Transform in Frequency Space
# This plot is similar to Plot 1 but for frequency space. It compares the inverse Fourier transform in frequency space
# with the original Gaussian.
plt.figure(2)
plt.plot(x, fn_f.real, 'bo', label='Re(fn)')  # Plot the real part of the inverse Fourier transform (fn_f)
plt.plot(x, fn_f.imag, 'g', label='Im(fn)')  # Plot the imaginary part of the inverse Fourier transform (fn_f)
plt.plot(x, fa, '--k', label='fa')  # Plot the original Gaussian function for comparison
plt.title('Frequency Space: Numerical Inversion of Analytical FT')  # Title indicating this is the inverse FT in frequency space
plt.xlabel('x')  # Label for the x-axis
plt.ylabel('f')  # Label for the y-axis
plt.legend()  # Show legend to identify the plots
plt.axis([-10, 10, 0, 0.6])  # Set the axis limits for better visualization

# Plot 3: Forward Fourier Transform in Pulsation Space
# This plot shows the forward Fourier transform of the Gaussian in real space and compares it with its analytical form in pulsation space.
plt.figure(3)
plt.plot(xi, Fn.real, 'ko', label='Re(Fn)')  # Plot the real part of the Fourier transform (Fn)
plt.plot(xi, Fn.imag, 'm', label='Im(Fn)')  # Plot the imaginary part of the Fourier transform (Fn)
plt.plot(xi, Fa_p, '--r', label='Fa_p')  # Plot the analytical Gaussian in pulsation space for comparison
plt.title('Xi Space - Forward Numerical FT of Analytical Expression')  # Title indicating this is the FT in pulsation space
plt.xlabel('xi')  # Label for the xi-axis
plt.ylabel('F')  # Label for the F-axis
plt.legend()  # Show legend to identify the plots
plt.axis([-20, 20, 0, 1.1])  # Set the axis limits for better visualization

# Plot 4: Forward Fourier Transform in Frequency Space
# This plot is similar to Plot 3 but for frequency space. It shows the forward Fourier transform of the Gaussian
# in real space and compares it with its analytical form in frequency space.
plt.figure(4)
plt.plot(nu, Fn.real, 'ko', label='Re(Fn)')  # Plot the real part of the Fourier transform (Fn)
plt.plot(nu, Fn.imag, 'm', label='Im(Fn)')  # Plot the imaginary part of the Fourier transform (Fn)
plt.plot(nu, Fa_f, '--b', label='Fa_f')  # Plot the analytical Gaussian in frequency space for comparison
plt.title('Nu Space - Forward Numerical FT of Analytical Expression')  # Title indicating this is the FT in frequency space
plt.xlabel('nu')  # Label for the nu-axis
plt.ylabel('F')  # Label for the F-axis
plt.legend()  # Show legend to identify the plots
plt.axis([-10/np.pi, 10/np.pi, 0, 1.1])  # Set the axis limits for better visualization

plt.show()