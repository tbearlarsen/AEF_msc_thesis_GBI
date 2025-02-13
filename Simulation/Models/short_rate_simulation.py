import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
====================================================================================================
Loading the data
====================================================================================================
"""

rates = pd.read_excel(r"/Users/osito/Library/CloudStorage/OneDrive-CBS-CopenhagenBusinessSchool/Masters/3. Semester/Library/AEF_msc_thesis_GBI/Simulation/Data/short_rate.xlsx", parse_dates=[0], index_col=0)


"""
====================================================================================================
Estimating the Vasicek parameters
====================================================================================================
"""
#Define the time step for daily data
dpa=220 #days per annum
dt = 1/dpa

# Create lagged series: r_t (current week) and r_t1 (next week)
r_t = rates[:-1]   # all observations except the last one
r_t1 = rates[1:]   # all observations except the first one

# Convert to numpy arrays for the regression
r_t = r_t.values
r_t1 = r_t1.values

# Prepare the independent variable for OLS by adding a constant (for the intercept)
X = sm.add_constant(r_t)

# Run OLS regression: r_{t+1} = theta + phi * r_t + error
model = sm.OLS(r_t1, X).fit()
print(model.summary())

# Extract estimated coefficients:
theta_hat = model.params[0]  # the intercept (θ)
phi_hat   = model.params[1]  # the coefficient on r_t (φ)

# Recover continuous-time parameters:
# Since φ = exp(-a * dt), then:
a_hat=-np.log(phi_hat)/dt

# Given θ = b * (1 - φ), then:
b_hat=theta_hat/(1-phi_hat)

# Estimate σ_η from the regression residuals:
sigma_eta_hat = np.std(model.resid, ddof=1)  # standard deviation of the residuals

# Recover the volatility σ:
sigma_hat = sigma_eta_hat * np.sqrt(2 * a_hat / (1 - phi_hat**2))

# Display the estimated parameters:
print(f"Estimated a (annualized mean reversion speed): {a_hat}")
print(f"Estimated b (long-term mean level): {b_hat}")
print(f"Estimated sigma (annualized volatility): {sigma_hat}")


"""
====================================================================================================
Simulating the Short Rate 10 years into the future
====================================================================================================
"""
#Simulation settings for daily data
years_simulated = 10
num_days = int(years_simulated * dpa)
num_paths = 10000   # Number of Monte Carlo paths

#Initial short rate
r0 = rates.iloc[-1, 0] #last value of DESTR in the dataset

#Pre-calculate constants for efficiency
exp_a_dt = np.exp(-a_hat * dt)
sigma_dt = sigma_hat * np.sqrt((1 - np.exp(-2 * a_hat * dt)) / (2 * a_hat))

#Initialize simulation array: rows = time steps, columns = simulation paths
simulated_rates = np.zeros((num_days + 1, num_paths))
simulated_rates[0, :] = r0

# Monte Carlo simulation over the 10-year horizon
for t in range(1, num_days + 1):
    eps = np.random.normal(size=num_paths)
    simulated_rates[t, :] = (b_hat +
                             (simulated_rates[t - 1, :] - b_hat) * exp_a_dt +
                             sigma_dt * eps)


# Total simulation horizon (e.g., 40 years)
total_years = 40
num_days_total = int(total_years * dpa)  # dpa = days per annum

# Create an extended array for the full horizon
extended_rates = np.zeros((num_days_total + 1, num_paths))

# First 10 years: use the simulated rates
extended_rates[:num_days + 1, :] = simulated_rates

# Define a transition period (e.g., 5 years) during which rates revert from the last simulated value to b_hat
transition_years = 5
transition_days = int(transition_years * dpa)
start_transition = num_days  # end of the simulated period (10 years)
end_transition = start_transition + transition_days

# For each day during the transition, interpolate linearly between the last simulated rate and b_hat
for t in range(start_transition + 1, end_transition + 1):
    # alpha goes from 0 (at t = start_transition) to 1 (at t = end_transition)
    alpha = (t - start_transition) / transition_days
    extended_rates[t, :] = (1 - alpha) * simulated_rates[-1, :] + alpha * b_hat

# Beyond the transition period, set the rate constant at b_hat
for t in range(end_transition + 1, num_days_total + 1):
    extended_rates[t, :] = b_hat

# Plot a few sample paths for the entire horizon
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(extended_rates[:, i], lw=1.5, label=f'Path {i+1}' if i < 10 else "")
plt.xlabel('Days')
plt.ylabel('Short Rate')
plt.title('Extended Short Rate Paths with Transition to Long-Run Mean')
plt.grid(True)
plt.show()





"""# For each path, the final rate after 10 years becomes the constant long-term rate.
# Here, we extend our simulation to the full horizon (e.g., 40 years) using this constant.
total_years = 40
num_days_total = int(total_years * 252)
extended_rates = np.zeros((num_days_total + 1, num_paths))
extended_rates[:num_days + 1, :] = simulated_rates

# Set the rate beyond 10 years to the last simulated value (per path)
for t in range(num_days + 1, num_days_total + 1):
    extended_rates[t, :] = simulated_rates[-1, :]

# Plot a few sample paths for the entire horizon
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(extended_rates[:, i], lw=1.5, label=f'Path {i+1}')
plt.xlabel('Days')
plt.ylabel('Short Rate')
plt.title('Extended Short Rate Paths (10-Year Simulation + Constant Long-Term Rate)')
plt.grid(True)
plt.show()"""




