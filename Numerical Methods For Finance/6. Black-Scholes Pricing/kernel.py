import numpy as np
from numpy.fft import fft, ifftshift, fftshift
import matplotlib.pyplot as plt
from charfunction import charfunction

def kernel(ngrid, xmin, xmax, parameters, alpha=0, disc=1, flag=0):
    """
    Computes the kernel for density estimation or option pricing.

    Args:
    ngrid (int): Number of grid points.
    xmin (float): Minimum value of the grid.
    xmax (float): Maximum value of the grid.
    parameters (dict): Parameters for the characteristic function.
    alpha (float, optional): Shift parameter, especially for Feng-Linetsky and convolution. Defaults to 0.
    disc (int, optional): Indicates if discount factor is included in the density (1 for yes, 0 for no). Defaults to 1.
    flag (int, optional): Indicates characteristic function type (0 for backward, 1 for forward). Defaults to 0.

    Returns:
    tuple: Returns the grid in real space (x), the kernel in real space (h), the grid in Fourier space (xi),
           and the characteristic function in Fourier space (H).
    """

    # Setting up the grid in real space
    N = ngrid // 2
    dx = (xmax - xmin) / ngrid
    x = dx * np.arange(-N, N)

    # Setting up the grid in Fourier space
    dxi = 2 * np.pi / (xmax - xmin)
    xi = dxi * np.arange(-N, N)

    # Computing the characteristic function
    H = charfunction(xi + 1j * alpha, parameters, flag)  # characteristic function

    # Applying discount factor if disc == 1
    if disc == 1:
        H *= np.exp(-parameters['rf'] * parameters['dt'])  # Discounting

    # Computing the kernel in real space using FFT
    h = np.real(fftshift(fft(ifftshift(H)))) / (xmax - xmin)

    # Use ctrl + / to comment or uncomment out rest of this
    # Plotting the real and imaginary parts of the characteristic function
    plt.figure()
    plt.plot(xi, H.real, 'r', xi, H.imag, 'g')
    plt.xlabel('xi')
    plt.ylabel('Psi(xi, Delta t)')
    plt.legend(['Re Psi(xi, Delta t)', 'Im Psi(xi, Delta t)'])
    plt.title('Characteristic function')
    plt.show()

    # Plotting the probability density function
    plt.figure()
    plt.plot(x, h)
    plt.xlabel('x')
    plt.ylabel('f(x, Delta t)')
    plt.title('Probability density function')
    plt.show()

    return x, h, xi, H