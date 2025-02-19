import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters and initial settings
# -------------------------------

S0 = 100.0         # Initial asset value
mu = 0.07          # Annual drift for GBM (7%)
sigma = 0.2        # Annual volatility (20%)
T1 = 10            # Stochastic simulation horizon in years
T2 = 35            # Deterministic growth horizon in years (total = 45 years)
g = 0.04           # Annual deterministic growth rate (4%)

# For the GBM simulation, we use monthly steps
steps_per_year = 12
n_steps = T1 * steps_per_year  # Total number of steps in the stochastic period
dt = T1 / n_steps              # Time increment in years
n_paths = 10000                # Number of Monte Carlo simulation paths

# -------------------------------
# Step 1: Simulate the 10-year stochastic period using GBM
# -------------------------------

# Initialise an array to hold the simulated asset paths
S = np.zeros((n_paths, n_steps + 1))
S[:, 0] = S0  # Set the initial asset value for all paths

# Loop over each time step and simulate the GBM process
for t in range(1, n_steps + 1):
    # Generate random standard normal shocks
    z = np.random.normal(0, 1, n_paths)
    # Discretised GBM formula:
    # S[t] = S[t-1] * exp[(mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z]
    S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

# At the end of T1 years, extract the terminal asset values
S_T1 = S[:, -1]

# -------------------------------
# Step 2: Apply deterministic growth for the remaining 35 years
# -------------------------------

# For each simulated path, project the asset value deterministically:
S_T45 = S_T1 * (1 + g) ** T2

# -------------------------------
# Step 3: Analysis & Visualisation
# -------------------------------

# Plot a histogram of the terminal asset values at 45 years
plt.figure(figsize=(10, 6))
plt.hist(S_T45, bins=50, edgecolor='black', alpha=0.7)
plt.title("Distribution of Asset Values at 45 Years")
plt.xlabel("Asset Value")
plt.ylabel("Frequency")
plt.show()