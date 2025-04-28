# File: config/experiment_config.py
class ExperimentConfig:
    def __init__(self):
        # 基本设置
        self.num_clients = 30
        self.num_rounds = 100
        self.dataset = "cifar10"
        # 客户端分组
        self.resource_groups = {
            "high":     {"ratio": 0.2,  "rci_range": [0.8, 1.0]},
            "medium":   {"ratio": 0.4,  "rci_range": [0.4, 0.8]},
            "low":      {"ratio": 0.35, "rci_range": [0.1, 0.4]},
            "very_low": {"ratio": 0.05, "rci_range": [0.05, 0.1]},
        }
        # 不稳定客户端设置
        self.unstable_ratio = 0.1
        self.participation_probability = 0.7
        # 算法超参数
        self.dfs_alpha = 0.5
        self.rci_weights = [0.25, 0.25, 0.25, 0.25]  # λ_C, λ_M, λ_E, λ_B
        self.distill_temp = 4.0

