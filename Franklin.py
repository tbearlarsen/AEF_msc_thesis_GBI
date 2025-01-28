import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ============================
# PARAMETERS
# ============================
N = 10  # Number of goals
k = 5  # Number of assets
varrho = 20  # Resolution level
tau = 1000  # Number of Monte Carlo simulations
W = np.random.uniform(50, 200, N)  # Random wealth levels for each goal
t = 5  # Time horizon
G = 1  # Scaling factor
V = np.random.uniform(0.5, 1.5, N)  # Simulated value functions (utility)

pi_func = lambda omega: (0, 1)  # Example: Mean=0, StdDev=1


# ============================
# STAGE 1: WITHIN-GOAL OPTIMIZATION
# ============================
def lower_tail_cdf(Wj, G, t, omega, pi_func):
    """Computes the lower-tail CDF function (probability of not achieving goal)."""
    result = norm.cdf((Wj / (G * np.dot(omega, np.ones(len(omega))))) ** (1 / t) - 1, *pi_func(omega))
    return float(result)  # Ensure scalar output


def objective_function(omega, Wj, G, t, pi_func):
    """Objective function for within-goal optimization (minimize downside risk)."""
    penalty = 100 * (1 - np.sum(omega)) ** 2  # Ensure weights sum to 1
    return lower_tail_cdf(Wj, G, t, omega, pi_func) + penalty


def weight_constraint(omega):
    """Constraint: portfolio weights sum to 1."""
    return np.sum(omega) - 1


# Bounds and constraints for portfolio optimization
bounds = [(0, 1) for _ in range(k)]
constraints = {"type": "eq", "fun": weight_constraint}

# Storage for optimal portfolios
Omega_matrices = np.zeros((N, varrho, k))

# Solve for optimal portfolios within each goal
for j in range(N):
    for i in range(1, varrho + 1):
        theta_i = i / varrho  # Compute resolution step
        omega_init = np.ones(k) / k  # Equal allocation starting point
        result = minimize(objective_function, omega_init, args=(W[j], G, t, pi_func),
                          bounds=bounds, constraints=constraints, method="SLSQP")
        Omega_matrices[j, i - 1, :] = result.x  # Store optimal portfolio weights


# ============================
# STAGE 2: ACROSS-GOAL OPTIMIZATION (Monte Carlo)
# ============================
def generate_theta(N):
    """Generates a feasible random across-goal allocation vector (sums to 1)."""
    return np.random.dirichlet(np.ones(N))


def objective_function_across_goals(theta, W, G, Omega_matrices, V, pi_func):
    """Objective function for across-goal Monte Carlo simulation."""
    total_value = 0
    for j in range(N):
        row_index = int(theta[j] * (varrho - 1))  # Select row in Omega
        investment_weights = Omega_matrices[j, row_index]
        failure_prob = lower_tail_cdf(W[j], G, t, investment_weights, pi_func)
        total_value += V[j] * (1 - failure_prob)  # Maximize utility
    return total_value  # Higher is better


# Monte Carlo Optimization
best_theta = None
best_value = -np.inf

for i in range(tau):
    theta_candidate = generate_theta(N)
    value = objective_function_across_goals(theta_candidate, W, G, Omega_matrices, V, pi_func)

    if value > best_value:
        best_value = value
        best_theta = theta_candidate

# Extract optimal investment weights for each goal based on best_theta
optimal_weights = np.zeros((N, k))
for j in range(N):
    row_index = int(best_theta[j] * (varrho - 1))  # Select optimal row
    optimal_weights[j, :] = Omega_matrices[j, row_index]

# ============================
# OUTPUT RESULTS
# ============================
df_theta = pd.DataFrame(best_theta, columns=["Optimal Theta"])
df_weights = pd.DataFrame(optimal_weights, columns=[f"Asset_{i + 1}" for i in range(k)])

# Save results to CSV files
df_theta.to_csv("optimal_theta.csv", index=False)
df_weights.to_csv("optimal_weights.csv", index=False)

# Print results
print("Optimal Theta (Wealth Allocations Across Goals):")
print(df_theta.head())

print("\nOptimal Investment Weights for Each Goal:")
print(df_weights.head())
