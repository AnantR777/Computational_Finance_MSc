import numpy as np
from scipy.special import gamma


def charfunction(xi, parameters, flag=0):
    """
    # charfunction: Wrapper function to compute the characteristic function with mean correction
    # and optional conjugation, used in financial models for various probability distributions.
    #
    # Inputs:
    # xi: Points in Fourier space where the characteristic function is evaluated.
    # parameters: Dictionary containing distribution parameters and other relevant values.
    # flag (optional): Indicator for backward (0) or forward (1) characteristic function. Defaults to 0.
    #
    # Outputs:
    # F: Evaluated characteristic function at points xi with mean correction and optional conjugation.
    #
    # Usage:
    # This function is used for advanced financial models where a risk-neutral valuation framework is applied.
    # It adjusts the characteristic function calculated by charfunction0 with a mean correction to align it
    # with the risk-free rate and dividend yield, and applies conjugation based on the specified flag.
    """

    if 'flag' not in parameters:
        flag = 0

    # Mean correction term, used to adjust for drift in the risk-neutral measure
    meancorrection = (parameters['rf'] - parameters['q']) * parameters[
        'dt'] - np.log(charfunction0(-1j, parameters))

    # Initial characteristic function computation
    F = np.exp(1j * meancorrection * xi) * charfunction0(xi, parameters)

    # Conjugate the function for backward problems (flag=0)
    if flag == 0:
        F = np.conj(F)

    return F


def charfunction0(xi, parameters):
    """
    # charfunction0: Base function to compute the characteristic function for specified distributions.
    #
    # Inputs:
    # xi: Points in Fourier space.
    # parameters: Dictionary containing distribution parameters.
    #
    # Outputs:
    # F: Base characteristic function of the specified distribution.
    #
    # Usage:
    # This function computes the core characteristic function for various probability distributions
    # such as Normal, Normal Inverse Gaussian (NIG), Variance Gamma (VG), and others. It is called
    # within charfunction to provide the fundamental characteristic function, without mean correction
    # or conjugation adjustments.
    """

    # Selecting the distribution and computing its characteristic function
    if parameters['distr'] == 1:  # Normal distribution
        m = parameters['m']
        s = parameters['s']
        F = np.exp(1j * xi * m - 0.5 * (s * xi) ** 2)

    elif parameters['distr'] == 2:  # Normal Inverse Gaussian (NIG)
        alpha = parameters['alpha']
        beta = parameters['beta']
        delta = parameters['delta']
        F = np.exp(-delta * (
                    np.sqrt(alpha ** 2 - (beta + 1j * xi) ** 2) - np.sqrt(
                alpha ** 2 - beta ** 2)))

    elif parameters['distr'] == 3:  # Variance Gamma (VG)
        theta = parameters['theta']
        s = parameters['s']
        nu = parameters['nu']
        F = (1 - 1j * xi * theta * nu + 0.5 * nu * (s * xi) ** 2) ** (-1 / nu)

    elif parameters['distr'] == 4:  # Meixner
        alpha = parameters['alpha']
        beta = parameters['beta']
        delta = parameters['delta']
        F = (np.cos(beta / 2) / np.cosh((alpha * xi - 1j * beta) / 2)) ** (
                    2 * delta)

    elif parameters['distr'] == 5:  # CGMY
        C = parameters['C']
        G = parameters['G']
        M = parameters['M']
        Y = parameters['Y']
        F = np.exp(C * gamma(-Y) * (
                    (M - 1j * xi) ** Y - M ** Y + (G + 1j * xi) ** Y - G ** Y))

    elif parameters['distr'] == 6:  # Kou
        s = parameters['s']
        lambda_ = parameters['lambda']
        pigr = parameters['pigr']
        eta1 = parameters['eta1']
        eta2 = parameters['eta2']
        F = np.exp(-0.5 * (s * xi) ** 2 + lambda_ * (
                    (1 - pigr) * eta2 / (eta2 + 1j * xi) + pigr * eta1 / (
                        eta1 - 1j * xi) - 1))

    elif parameters['distr'] == 7:  # Merton
        s = parameters['s']
        alpha = parameters['alpha']
        lambda_ = parameters['lambda']
        delta = parameters['delta']
        F = np.exp(-0.5 * (s * xi) ** 2 + lambda_ * (
                    np.exp(1j * xi * alpha - 0.5 * (delta * xi) ** 2) - 1))

    elif parameters['distr'] == 8:  # Levy alpha-stable
        alpha = parameters['alpha']
        beta = parameters['beta']
        gamm = parameters['gamm']
        m = parameters['m']
        c = parameters['c']
        F = np.exp(1j * xi * m - c * abs(gamm * xi) ** alpha * (
                    1 - 1j * beta * np.sign(xi) * np.tan(alpha / 2 * np.pi)))

    else:
        raise ValueError("Invalid distribution type specified.")

    return F