import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os


class VasicekModel:
    def __init__(self, data_file, periods_per_annum=252):
        """
        Initialise the Vasicek model.

        Parameters:
        data_file : str
            Path to the Excel file containing the rate data.
        periods_per_annum : int, optional
            Number of observations per year (default is 252, e.g. trading days).
        """
        self.data_file = data_file
        self.periods_per_annum = periods_per_annum
        self.dt = 1 / periods_per_annum
        self.rates = self.load_data()
        self.a_hat = None
        self.b_hat = None
        self.sigma_hat = None

    def load_data(self):
        """Load and clean short rate data from an Excel file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {self.data_file}")
        rates = pd.read_excel(self.data_file, parse_dates=[0], index_col=0)
        return rates.dropna()

    def estimate_params(self):
        """Estimate Vasicek model parameters using OLS regression."""
        # Create lagged series
        r_t = self.rates[:-1].values
        r_t1 = self.rates[1:].values

        # OLS regression: r_{t+1} = theta + phi * r_t + error
        X = sm.add_constant(r_t)
        model = sm.OLS(r_t1, X).fit()
        print(model.summary())

        theta_hat = model.params[0]
        phi_hat = model.params[1]

        # Recover continuous-time parameters
        self.a_hat = -np.log(phi_hat) / self.dt
        self.b_hat = theta_hat / (1 - phi_hat)

        # Estimate volatility
        sigma_eta_hat = np.std(model.resid, ddof=1)
        self.sigma_hat = sigma_eta_hat * np.sqrt(2 * self.a_hat / (1 - phi_hat ** 2))

        print(f"Estimated a: {self.a_hat}")
        print(f"Estimated b: {self.b_hat}")
        print(f"Estimated sigma: {self.sigma_hat}")

    def simulate(self, r0, years, num_paths):
        """Simulate short rate paths over a specified number of years."""
        num_periods = int(years * self.periods_per_annum)
        exp_a_dt = np.exp(-self.a_hat * self.dt)
        sigma_dt = self.sigma_hat * np.sqrt((1 - np.exp(-2 * self.a_hat * self.dt)) / (2 * self.a_hat))

        sim_rates = np.zeros((num_periods + 1, num_paths))
        sim_rates[0, :] = r0

        for t in range(1, num_periods + 1):
            eps = np.random.normal(size=num_paths)
            sim_rates[t, :] = (self.b_hat +
                               (sim_rates[t - 1, :] - self.b_hat) * exp_a_dt +
                               sigma_dt * eps)
        return sim_rates

    def extend_simulation(self, simulated_rates, total_years, k=3, c=0.5):
        """Extend the simulation with enhanced mean reversion parameters."""
        num_periods_sim = simulated_rates.shape[0] - 1
        num_paths = simulated_rates.shape[1]
        total_periods = int(total_years * self.periods_per_annum)
        extended_rates = np.zeros((total_periods + 1, num_paths))
        extended_rates[:num_periods_sim + 1, :] = simulated_rates

        # Modified parameters for extended simulation
        a_long = self.a_hat * k
        sigma_long = self.sigma_hat * c

        exp_a_dt_long = np.exp(-a_long * self.dt)
        sigma_dt_long = sigma_long * np.sqrt((1 - np.exp(-2 * a_long * self.dt)) / (2 * a_long))

        for t in range(num_periods_sim + 1, total_periods + 1):
            eps = np.random.normal(size=num_paths)
            extended_rates[t, :] = (self.b_hat +
                                    (extended_rates[t - 1, :] - self.b_hat) * exp_a_dt_long +
                                    sigma_dt_long * eps)
        return extended_rates

    def plot_simulation(self, simulation, num_paths_to_plot=10, xlabel="Periods", ylabel="Short Rate",
                        title="Simulation Paths"):
        """Plot a sample of simulation paths."""
        plt.figure(figsize=(12, 6))
        for i in range(num_paths_to_plot):
            plt.plot(simulation[:, i], lw=1.5, label=f'Path {i + 1}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()