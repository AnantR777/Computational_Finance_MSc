import numpy as np
import matplotlib.pyplot as plt
import time

# ===========================================
#  Rectangular Unit Pulse <--> Sinc function
# ===========================================

# This script demonstrates the Fourier transformation of a rectangular pulse (in real space)
# to a sinc function (in Fourier space). It uses both Fast Fourier Transform (FFT) and
# Discrete Fourier Transform (DFT) for the transformation and provides a comparison between them.

# To help us understand how the FFT algorithm works, we can perform the
# fourier transform on functions that we already have an analytical
# solution to and then compare.

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
N = 512 # Number of grid points - determines the resolution in both real and Fourier spaces
dx = 0.1 # Grid step in real space - spacing between individual points
upperx = N*dx # Upper truncation limit in real space - defines the total span in real space
x = dx * np.arange(-N / 2, N / 2)  # Grid in real space - array of evenly spaced points

# GRID IN FOURIER SPACE (Pulsation)
dxi = (2*np.pi)/(N*dx)  # Grid step in Fourier space - frequency resolution
upperxi = N*dxi # Upper truncation limit in fourier space
xi = dxi * np.arange(-N / 2, N / 2)  # Grid in Fourier space - array of frequency points

# GRID IN FOURIER SPACE (Frequency)
dnu = 1/(N*dx) #Grid step size in fourier space
uppernu = N*dnu #  Upper truncation limit in fourier space
nu = dnu * np.arange(-N / 2, N / 2)  # Grid in Fourier space

## STEP 2: Analytical Expressions & Functions

# Unit Pulse Function:
# f(x) = indicator[-a,a]

# Sinc function:
# f(xi) = 2*sin(a*xi)/xi             --- Pulsation
# f(nu) = 2*sin(2*a*pi*nu)/(2*pi*nu) --- Frequency

# ANALYTICAL expressions
# -----------------------------
a = 2 # Parameter

# Define the Unit Pulse Function
fa = np.ones(N)
fa[np.abs(x) > a] = 0

# Define the Sinc function in Pulsation space
Fa_p = 2 * np.sin(a * xi) / xi
Fa_p[N//2] = 2 * a  # Correction for NaN at xi=0 using L'Hopital's rule
# --Alternative formulation--
#Fa_p = (np.exp(1i*a*xi) - np.exp(-1i*a*xi))/xi;
#Fa_p[N//2] = 2*a

# Define the Sinc function in Frequency space
Fa_f = 2 * np.sin(2 * a * np.pi * nu) / (2 * np.pi * nu)
Fa_f[N//2] = 2 * a  # Correction for NaN at nu=0 using L'Hopital's rule
# --Alternative formulation--
#Fa_f = np.(exp(1i*a*2*pi*nu) - np.exp(-1i*a*2*pi*nu))/(2*np.pi*nu);
#Fa_f[N//2] = 2 * a

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

# Assume that the variables 'fa', 'x', 'xi', and 'dx' are already defined from your earlier code

# Start the timer to measure the execution time of the DFT process
start_time = time.time()

# Freq version (uniform-to-sine transformation)

# Initialize an array of zeros to store the DFT results
# The 'dtype=complex' ensures that the array can store complex numbers
Fn2_f = np.zeros(N, dtype=complex)

# Nested loop for the DFT calculation
for j in range(N):  # Iterating over each frequency bin 'j'
    for k in range(N):  # Iterating over each time point 'k' of the signal
        # Compute the DFT for the uniform-to-sine transformation
        # The complex exponential function is a key component of the Fourier transform
        Fn2_f[j] += np.exp(1j * 2 * np.pi * j * k / N) * fa[k] * dx

# Initialize an array of zeros for the IDFT results
fn2_f = np.zeros(N, dtype=complex)

# Nested loop for the IDFT calculation
for n in range(N):  # Iterating over each time point 'n'
    for k in range(N):  # Iterating over each frequency bin 'k'
        # Compute the IDFT for the uniform-to-sine transformation
        # The complex exponential has the opposite sign in the exponent compared to DFT
        fn2_f[n] += Fn2_f[k] * np.exp(-1j * 2 * np.pi * k * n / N)

# Normalize the IDFT result by dividing by the number of points 'N'
fn2_f /= N

# Calculate and print the DFT computation time
dft_time = time.time() - start_time
print("DFT time (uniform-to-sine): ", dft_time)

#######################################
# Pulsation version (uniform-to-sine transformation)

# Initialize an array of zeros to store the DFT results for the pulsation version
Fn2_p = np.zeros(N, dtype=complex)

# Nested loop to compute the DFT for the pulsation version
for j in range(N):  # Iterating over each frequency bin 'j'
    for k in range(N):  # Iterating over each time point 'k' of the signal
        # Compute the DFT for the uniform-to-sine transformation (pulsation version)
        Fn2_p[j] += np.exp(1j * xi[j] * x[k]) * fa[k] * dx

# Initialize an array of zeros for the IDFT results in the pulsation version
fn2_p = np.zeros(N, dtype=complex)

# Nested loop to compute the IDFT for the pulsation version
for n in range(N):  # Iterating over each time point 'n'
    for k in range(N):  # Iterating over each frequency bin 'k'
        # Compute the IDFT for the uniform-to-sine transformation (pulsation version)
        fn2_p[n] += Fn2_p[k] * np.exp(-1j * xi[k] * x[n])

# Normalize the IDFT result by dividing by the total time period
fn2_p /= N * dx


## STEP 4: Plotting

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
plt.axis([-10, 10, 0, 1.2])  # Set the axis limits for better visualization

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
plt.axis([-10, 10, 0, 1.2])  # Set the axis limits for better visualization

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
plt.axis([-20, 20, -a, 2.5*a])  # Set the axis limits for better visualization

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
plt.axis([-10/np.pi, 10/np.pi, -a, 2.5*a])  # Set the axis limits for better visualization


# Plotting the Results
# Plot 1: Rectangular Pulse and Its Fourier Transforms
plt.figure(5), plt.clf(), plt.grid(True)
plt.plot(x, fa, 'r', x, fn_p.real, 'og', x, fn_p.imag, '.g', x, fn2_p.real, 'xb', x, fn2_p.imag, ':b')
plt.axis([-4, 4, 0, 1.2])
plt.xlabel('x')
plt.ylabel('f')
plt.legend(['Analytic', 'Re IFFT(F)', 'Im IFFT(F)', 'Re IDFT(F)', 'Im IDFT(F)'])
plt.title('Rectangular Pulse and Its Fourier Transforms')

# Plot 2: Sinc Function and Its Inverse Fourier Transforms
plt.figure(6), plt.clf(), plt.grid(True)
plt.plot(xi, Fa_p, 'r', xi, Fn.real, 'og', xi, Fn.imag, '.g', xi, Fn2_p.real, 'xb', xi, Fn2_p.imag, ':b')
plt.axis([-20, 20, -2, 5])
plt.xlabel('ξ')
plt.ylabel('F')
plt.legend(['Analytic', 'Re FFT(f)', 'Im FFT(f)', 'Re DFT(f)', 'Im DFT(f)'])
plt.title('Sinc Function and Its Inverse Fourier Transforms')
plt.show()