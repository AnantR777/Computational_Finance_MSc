# ===============================================================
# Merton Jump-Diffusion Process
# X(t) = (muS-0.5*sigma^2)*t + sigma*W(t) + sum_{i=1}^{N(t)} Z_i
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class MertonJD(object):
    """
    Merton JD Process
    """

    def __init__(self, steps, T, paths, muS, sigma, muJ, sigmaJ, lbda, s0):
        """
        Initialize the Merton JD Process Model.

        Parameters:
        - steps (int): The number of time steps in the simulation.
        - time_horizon (float): The total time duration of the simulation.
        - paths (int): The number of paths to simulate.
        - muS (float); sigma (float): model parameters of the diffusion part (GBM)
        - muJ (float); sigmaJ (float); lbda: model parameters of the jump part (NCPP)
        - s0 (float): initial stock price
        """
        self.steps = steps
        self.T = T
        self.paths = paths
        self.dt = self.T / self.steps
        self.t = np.arange(0, self.T + self.dt, self.dt)
        self.muS = muS
        self.sigma = sigma
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.lbda = lbda
        self.s0 = s0
        # random number generator
        self.rng = np.random.default_rng()

    # ========================
    # Monte Carlo Simulation 
    # ========================

    def simulate_monte_carlo(self):
        """
        Returns:
        - numpy.ndarray: Simulated paths of the Merton JD Process.
        """
        # compute increments of the ABM
        randn = self.rng.normal(size=(self.steps, self.paths))
        dW = (
                         self.muS - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(
            self.dt) * randn

        # compute the incremets of the NCPP 
        dN = self.rng.poisson(self.lbda * self.dt,
                              size=(self.steps, self.paths))
        randn2 = self.rng.normal(size=(self.steps, self.paths))

        dJ = self.muJ * dN + self.sigmaJ * np.sqrt(dN) * randn2

        # sum the increments of the ABM and the NCPP 
        dX = dW + dJ

        # accumulate the increments    
        cumdX = np.cumsum(dX, 0)
        self.X = np.concatenate([np.zeros(shape=(1, self.paths)), cumdX])

        print('Simulated Merton JD Process with {} paths and {} steps.'.format(
            self.paths, self.steps))

    # ===================================
    # Fourier Transform, Characteristic Function
    # =================================== 

    def characteristic_function(self, u, t):
        # characteristic function of the ABM
        cf_abm = np.exp(
            1j * u * self.muS * t - 0.5 * u ** 2 * self.sigma ** 2 * t)
        # characteristic function of the NCPP
        cf_ncpp = np.exp(self.lbda * (np.exp(
            1j * u * self.muJ - 0.5 * u ** 2 * self.sigmaJ ** 2) - 1) * t)
        return cf_abm * cf_ncpp

    def fourier_transform_pdf(self, time_step):

        N = 1000  # number of grid points
        Dx = 0.1  # grid spacing
        Lx = N * Dx  # upper truncation limit in real space
        Dxi = 2 * np.pi / Lx  # spacing in Fourier space
        x = Dx * (np.arange(-N / 2, N / 2))  # real space grid
        xi = Dxi * (np.arange(-N / 2, N / 2))  # Fourier space grid
        Lxi = N * Dxi  # upper limit in Fourier space

        cf_values = self.characteristic_function(xi, time_step)

        fn1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(cf_values))) / Lx

        return fn1, x

    # ===================================
    # Moments, Calculations
    # =================================== 

    def expected_path(self):
        # expected = (mu + lambda*muJ) * t
        return (self.muS + self.lbda * self.muJ) * self.t

    def mean(self):
        return np.mean(self.X, 1)

    def variance(self):
        # variance = (sigma^2 + lambda(muJ^2 + sigmaJ^2)) * t 
        theory = (self.sigma ** 2 + self.lbda * (
                    self.muJ ** 2 * self.sigmaJ ** 2)) * self.t
        return np.var(self.X, 1), theory

    # ========================
    # Plotting Methods 
    # ========================   

    def plot_expected(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.t, self.expected_path(), label='Expected Path',
                 linestyle='dashed', linewidth=1.5)
        plt.plot(self.t, self.mean(), label='Mean', linestyle='dotted')
        plt.plot(self.t, self.X[:, 0:10], linewidth=0.5)
        plt.xlabel('Time')
        plt.ylabel('X(t)')
        plt.legend()
        plt.title('Paths of a Merton Jump Diffusion Process')
        # plt.savefig('mjd_mean.png')
        plt.show()

    def plot_variance(self):
        plt.figure(figsize=(8, 6))
        actual_variance, theoretical_variance = self.variance()
        plt.plot(self.t, theoretical_variance, label='Theoretical Variance')
        plt.plot(self.t, actual_variance, label='Actual Variance')
        plt.xlabel('Time')
        plt.ylabel('Variance = E((X(t) - E(X(t)))^2)')
        plt.legend()
        plt.title('Variance of a Merton Jump Diffusion Process')
        # plt.savefig('mjd_variance.png')
        plt.show()

    def plot_pdfs(self):
        # Probability Density Function at Different Times
        plt.subplots(3, 1, figsize=(8, 8))
        bin_edges = 'auto'
        x_min = min(self.X[200, :]) - 1
        x_max = max(self.X[200, :]) + 2

        timings = [0.2, 0.4, 1.0]
        time_steps = [int(fraction * self.steps) for fraction in timings]

        for i, time_step in enumerate(time_steps, 1):
            plt.subplot(3, 1, i)
            xdata = self.X[time_step, :]

            plt.hist(xdata, bins=bin_edges, edgecolor='black', density=True,
                     alpha=0.7)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title(
                f'PDF of a Merton JD Process at time {time_step * self.dt}',
                fontsize=10)
            plt.ylabel(f'f_X(x,{time_step * self.dt})')
            plt.xlim(x_min, x_max)

        plt.legend()
        # plt.savefig('mjd_pdf.png')
        plt.xlabel('x', fontsize=8)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def plot_fourier_and_simulated_pdfs(self):
        # Probability Density Function at Different Times
        plt.subplots(3, 1, figsize=(8, 8))
        bin_edges = 'auto'
        x_min = min(self.X[200, :]) - 1
        x_max = max(self.X[200, :]) + 2

        timings = [0.2, 0.4, 1.0]
        time_steps = [int(fraction * self.steps) for fraction in timings]

        for i, time_step in enumerate(time_steps, 1):
            plt.subplot(3, 1, i)
            xdata = self.X[time_step, :]

            plt.hist(xdata, bins=bin_edges, edgecolor='black', density=True,
                     alpha=0.7)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.title(
                f'PDF of a Merton JD Process at time {time_step * self.dt}',
                fontsize=10)
            plt.ylabel(f'f_X(x,{time_step * self.dt})')
            plt.xlim(x_min, x_max)

            # Fourier Transform
            fn1, x = self.fourier_transform_pdf(time_step * self.dt)
            plt.plot(x, np.real(fn1), marker='*', label='real')
            plt.plot(x, np.imag(fn1), marker='o', label='imag')

        plt.legend()
        plt.title('PDF of a Merton JD Process')
        # plt.savefig('mjd_pdf.png')
        plt.xlabel('x', fontsize=8)
        plt.subplots_adjust(hspace=0.5)
        plt.show() 


mjd1 = MertonJD(600,1,20000,0.2,0.2,-0.1,0.15,0.05,1)
mjd1.simulate_monte_carlo()
mjd1.plot_expected()
mjd1.plot_variance()
mjd1.plot_pdfs()
mjd1.plot_fourier_and_simulated_pdfs()