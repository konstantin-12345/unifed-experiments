# utils.py
import numpy as np

def compute_rci(C, M, E, B, C_max, M_max, E_max, B_max, lambdas):
    λ_C, λ_M, λ_E, λ_B = lambdas
    return (
        λ_C * (C / C_max)
      + λ_M * (M / M_max)
      + λ_E * (1 - E / E_max)
      + λ_B * (B / B_max)
    )

def compute_dfs(delta, freq, alpha):
    # delta: L_i - L_mean; freq: participation frequency
    return alpha * sigmoid(-delta) + (1 - alpha) * (1 - freq)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
