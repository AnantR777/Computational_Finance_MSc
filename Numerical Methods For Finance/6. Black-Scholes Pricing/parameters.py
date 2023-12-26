import numpy as np
from scipy.special import gamma


def parameters(distr, T, dt, rf, q=0):
    """
    Sets process parameters for different probability distributions as used in financial models.

    Args:
    distr (int): Identifier for the chosen probability distribution.
    T (float): Total time period.
    dt (float): Time step.
    rf (float): Risk-free interest rate.
    q (float, optional): Dividend yield. Defaults to 0 if not provided.

    Returns:
    dict: A dictionary containing the parameters set for the specified distribution.
    """

    # Initialize a dictionary to store parameters
    params = {
        'distr': distr,
        'T': T,
        'dt': dt,
        'rf': rf,
        'q': q
    }

    # Define parameters based on the selected distribution
    if distr == 1:  # Normal
        m = 0  # Mean
        s = 0.4  # Standard deviation, can be adjusted

        # Rearrange parameters (time rescaling)
        params['m'] = m
        params['s'] = s * np.sqrt(dt)

        # Analyticity bounds
        params['lambdam'] = 0
        params['lambdap'] = 0

    elif distr == 2:  # NIG (Normal Inverse Gaussian)
        alpha, beta, delta = 15, -5, 0.5

        # Rearrange parameters (time rescaling)
        params['alpha'] = alpha
        params['beta'] = beta
        params['delta'] = delta * dt

        # Analyticity bounds
        params['lambdam'] = beta - alpha
        params['lambdap'] = beta + alpha

        # Additional parameters for grid calculation
        params['FLc'] = delta
        params['FLnu'] = 1

    # Continued from the previous elif block for distributions 1 and 2

    elif distr == 3:  # Variance Gamma (VG)
        C, G, M = 4, 12, 18

        nu = 1 / C
        theta = (1 / M - 1 / G) * C
        s = np.sqrt(2 * C / (M * G))

        # Rearrange parameters (time rescaling)
        params['nu'] = nu / dt
        params['theta'] = theta * dt
        params['s'] = s * np.sqrt(dt)

        # Analyticity bounds
        params['lambdam'] = -M
        params['lambdap'] = G

    elif distr == 4:  # Meixner
        alpha, beta, delta = 0.3977, -1.4940, 0.3462

        # Rearrange parameters (time rescaling)
        params['alpha'] = alpha
        params['beta'] = beta
        params['delta'] = delta * dt

        # Compute s for vol(T)
        s = alpha * np.sqrt(delta) / (4 * np.cos(beta / 2))

    elif distr == 5:  # CGMY (Carr, Geman, Madan, Yor)
        C, G, M, Y = 4, 50, 60, 0.7

        # Rearrange parameters (time rescaling)
        params['C'] = C * dt
        params['G'] = G
        params['M'] = M
        params['Y'] = Y

        # Analyticity bounds
        params['lambdam'] = -M
        params['lambdap'] = G

        # F&L grid parameters
        params['FLc'] = 2 * C * abs(gamma(-Y) * np.cos(np.pi * Y / 2))
        params['FLnu'] = Y

    elif distr == 6:  # Kou double exponential jump-diffusion
        s, lambda_, pigr, eta1, eta2 = 0.1, 3, 0.3, 40, 12

        # Rearrange parameters (time rescaling)
        params['s'] = s * np.sqrt(dt)
        params['lambda'] = lambda_ * dt
        params['pigr'] = pigr
        params['eta1'] = eta1
        params['eta2'] = eta2

        # Analyticity bounds
        params['lambdam'] = -eta1
        params['lambdap'] = eta2

        # F&L grid parameters
        params['FLc'] = s ** 2 / 2
        params['FLnu'] = 2

    elif distr == 7:  # Merton jump-diffusion
        s, alpha, lambda_, delta = 0.4, 0.1, 0.5, 0.15

        # Rearrange parameters (time rescaling)
        params['s'] = s * np.sqrt(dt)
        params['alpha'] = alpha
        params['lambda'] = lambda_ * dt
        params['delta'] = delta

        # Analyticity bounds
        params['lambdam'] = 0
        params['lambdap'] = 0

        # F&L grid parameters
        params['FLc'] = s ** 2 / 2
        params['FLnu'] = 2

    elif distr == 8:  # Levy alpha-stable
        alpha, beta, gamm, m, c = 2, 0, 0.3 / np.sqrt(2), 0, 1 / np.sqrt(1 + (beta * np.tan(np.pi * alpha / 2)) ** 2)

        # Rearrange parameters (time rescaling)
        params['alpha'] = alpha
        params['beta'] = beta
        params['gamm'] = gamm * dt
        params['m'] = m
        params['c'] = c

        # Compute s for vol(T)
        s = np.sqrt(2) * gamm if alpha == 2 else np.inf

    else:
        raise ValueError("Invalid distribution type specified.")

    return params
