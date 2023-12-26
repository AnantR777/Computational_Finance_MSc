import numpy as np
import matplotlib.pyplot as plt

def payoff(x, xi, alpha, K, L, U, C, theta):
    """
    Computes the scaled payoff and its Fourier transform for options.

    Parameters:
    x (numpy array):  Represents a grid in real space, likely a range of values for the logarithm of stock prices.
    xi (numpy array): Grid in Fourier space, used for the Fourier transform.
    alpha (float):  A damping factor used in the Fourier transform to ensure that the integral converges.
    K (float): Strike price of the option.
    L (float): Lower barrier for the option payoff.
    U (float): Upper barrier for the option payoff.
    C (float): Scaling factor for the asset price. like initital price
    theta (int): Option type indicator (1 for call, -1 for put).

    Returns:
    S (numpy array): Scaled asset prices.
    g (numpy array): Scaled payoff function in real space.
    G (numpy array): Fourier transform of the scaled payoff function.
    """

    # Scale the asset prices based on the real space grid x.
    S = C * np.exp(x)

    # Determine the analytical Fourier transform of the payoff.
    # Convert the barriers and strike price into log space.
    # range within which the option payoff is considered between l and u
    l = np.log(L / C)  # Lower log barrier
    k = np.log(K / C)  # Log strike
    u = np.log(U / C)  # Upper log barrier

    # Initialize the payoff based on option type (call or put).
    if theta == 1:  # Call option
        net_pay = theta * (
                    S - K)  # calculates the intrinsic value of the call option for each possible asset price S
        net = len(net_pay) - len(net_pay[
                                     net_pay > 0])  # calculates number of negative elements i.e when S < K
        g = np.concatenate((np.zeros(net), net_pay[net_pay > 0])) * (S >= L) * (
                    S <= U) * np.exp(alpha * x)  # produce real payoff

        # set the integration bounds for the Fourier transform. For a call option, the relevant range is between
        a = max(l,
                k)  # the max of the lower barrier (l) and strike price (k) and
        b = u  # the upper barrier (u)

    else:  # Put option
        net_pay = theta * (
                    S - K)  # calculates the intrinsic value of the put option for each possible asset price S
        net_pay[
            net_pay < 0] = 0  # calculates number of negative elements i.e when S > K
        g = net_pay * (S >= L) * (S <= U) * np.exp(
            alpha * x)  # produce real payoff

        # set the integration bounds for the Fourier transform. For a put option, the relevant range is between
        a = min(k,
                u)  # the min of the upper barrier (u) and strike price (k) and
        b = l  # the lower barrier (l)

    # Calculate the Fourier transform of the payoff function.

    # 1j is imaginary number in python. this line creates an imaginary number for each xi
    xi2 = alpha + 1j * xi

    # compute fourier transform
    G = C * ((np.exp(b * (1 + xi2)) - np.exp(a * (1 + xi2))) / (1 + xi2) - (
                np.exp(k + b * xi2) - np.exp(k + a * xi2)) / xi2)

    # Handle discontinuities at xi=0.
    xi_zero_index = int(np.floor(len(G) / 2))
    if alpha == 0:
        G[xi_zero_index] = C * (np.exp(b) - np.exp(a) - np.exp(k) * (b - a))
    elif alpha == -1:
        G[xi_zero_index] = C * (b - a + np.exp(k - b) - np.exp(k - a))

    # Use highlight and ctrl + / to comment or uncomment out rest of this
    # Computing the numerical payoff
    gn = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(G))) / (
                (x[1] - x[0]) * len(x))

    # Plotting
    plt.figure()
    plt.plot(x, g, 'g', x, gn.real, 'r')
    plt.xlabel('x')
    plt.ylabel('g')
    plt.legend(['analytical', 'numerical'])
    if theta == 1:
        plt.title('Damped payoff function for a call option')
    else:  # put option
        plt.title('Damped payoff function for a put option')
    plt.show()

    # Return the scaled asset prices, scaled payoff in real space, and its Fourier transform.
    return S, g, G

# Example usage of the payoff function:
# x, xi, alpha, K, L, U, C, theta = [your input values here]
# S, g, G = payoff(x, xi, alpha, K, L, U, C, theta)

# The function 'payoff' is designed to compute the payoff function for options and
# its Fourier transform, which are crucial in pricing options using Fourier transform techniques.