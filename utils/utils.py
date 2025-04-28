import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 计算资源复合指数 RCI
def compute_rci(C, M, E, B, C_max, M_max, E_max, B_max, lambdas):
    λ_C, λ_M, λ_E, λ_B = lambdas
    return (
        λ_C * (C / C_max)
      + λ_M * (M / M_max)
      + λ_E * (1 - E / E_max)
      + λ_B * (B / B_max)
    )

# 计算动态公平度量 DFS
def compute_dfs(delta, freq, alpha):
    return alpha * sigmoid(-delta) + (1 - alpha) * (1 - freq)
