# strategy.py
import flwr as fl
from utils import compute_rci, compute_dfs

class UniFedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 可在此设定超参，如 lambdas, alpha, distill_T 等

    def configure_fit(self, server_round, parameters, client_manager):
        # 1. 收集客户端资源与 DFS（通过 client_manager 或上轮 metrics）
        # 2. 计算本地迭代轮次 E_i、蒸馏温度等 hyperparams
        # 3. 返回 FitIns(parameters, config={...})
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # 1. 从 results 中提取 Δ_i、p_i、RCI_i、δ_i、n_i
        # 2. 按伪码中 w_i ∝ n_i·RCI_i·sigmoid(−δ_i) 计算聚合权重
        # 3. 合并参数 & 更新全局软标签
        return super().aggregate_fit(server_round, results, failures)
