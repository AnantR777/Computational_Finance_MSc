import matplotlib.pyplot as plt
import numpy as np


def figures_ft(S, x, xi, f, Psi, g, G):
    """
    Plots various aspects of the Fourier transform process in financial modeling.

    Args:
    S (array): Underlying asset prices.
    x (array): Grid in real space.
    xi (array): Grid in Fourier space.
    f (array): Probability density function in real space.
    Psi (array): Characteristic function in Fourier space.
    g (array): Payoff function in real space.
    G (array): Fourier transform of the payoff function.
    """

    # Plotting the characteristic function at time T
    plt.figure()
    plt.plot(xi, Psi.real, 'r', xi, Psi.imag, 'g')
    plt.xlabel('xi')
    plt.ylabel('Psi(xi,T)')
    plt.legend(['Re Psi(xi,T)', 'Im Psi(xi,T)'])
    plt.title('Characteristic function at time T')

    # Plotting the probability density function at time T
    plt.figure()
    plt.plot(x, f.real, 'r', x, f.imag, 'g')
    plt.xlabel('x')
    plt.ylabel('f(x,T)')
    plt.title('Probability density function at time T')

    # Comparison of the numerical and analytical Fourier transforms of the payoff

    # Normal space
    gn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G))) / ((x[1] - x[0]) * len(x))

    # Plotting the payoff function in real space
    plt.figure()
    plt.plot(x, g, 'r', x, gn.real, 'g')
    plt.xlabel('x')
    plt.ylabel('Re g')
    plt.legend(['analytical', 'numerical'])
    plt.title('Payoff function')

    plt.figure()
    plt.plot(x, np.zeros_like(x), 'ro', x, gn.imag, 'gs')
    plt.xlabel('x')
    plt.ylabel('Im g')
    plt.legend(['analytical', 'numerical'])
    plt.title('Payoff function')

    plt.figure()
    plt.plot(x, g / gn.real, 'gs')
    plt.xlabel('x')
    plt.ylabel('g_a/Re g_n')
    plt.xlim([0, S[-1]])  # Assuming S is sorted and has the upper limit
    plt.title('Payoff function')

    plt.figure()
    plt.plot(x, g, 'r', S, gn.real, ':r', x, gn.imag, ':g')
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['g_a', 'Re g_n', 'Im g_n'])
    plt.title('Payoff function')

    # Reciprocal space
    Gn = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(g))) * (x[1] - x[0]) * len(x)

    # Plotting the Fourier transform of the payoff function in reciprocal space
    plt.figure()
    plt.plot(xi, G.real, 'ro', xi, Gn.real, 'gs')
    plt.xlim([-50, 50])
    plt.xlabel('xi')
    plt.ylabel('Re G')
    plt.legend(['analytical', 'numerical'])
    plt.title('Fourier transform of the payoff function')

    plt.figure()
    plt.plot(xi, G.imag, 'ro', xi, Gn.imag, 'gs')
    plt.xlim([-50, 50])
    plt.xlabel('xi')
    plt.ylabel('Im G')
    plt.legend(['analytical', 'numerical'])
    plt.title('Fourier transform of the payoff function')

    plt.figure()
    plt.plot(xi, G.real / Gn.real, 'ro', xi, G.imag / Gn.imag, 'gs')
    plt.xlim([-50, 50])
    plt.xlabel('xi')
    plt.legend(['Re(G_a)/Re(G_n)', 'Im(G_a)/Im(G_n)'])
    plt.title('Fourier transform of the payoff function')

    plt.figure()
    plt.plot(xi, G.real, 'r', xi, Gn.real, ':r', xi, G.imag, 'g', xi, Gn.imag, ':g')
    plt.xlim([-50, 50])
    plt.xlabel('xi')
    plt.ylabel('G')
    plt.legend(['Re G_a', 'Re G_n', 'Im G_a', 'Im G_n'])
    plt.title('Fourier transform of the payoff function')

    plt.show()
