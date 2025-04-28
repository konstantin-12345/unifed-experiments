# File: strategy.py
import flwr as fl
import numpy as np
from utils.utils import compute_rci, compute_dfs
from config.experiment_config import ExperimentConfig
from utils.checkpoint import save_experiment_state

class UniFedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = ExperimentConfig()
        self.participation_history = {}
        self.participation_freq = {}
        self.global_history = []

    def configure_fit(self, server_round, parameters, client_manager):
        # 根据历史 metrics 计算 DFS
        # 此处可扩展：读取上轮 metrics 计算 delta, freq
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # 更新参与历史
        for (client_params, num_examples, metrics) in results:
            cid = metrics.get("client_id")
            self.participation_history[cid] = self.participation_history.get(cid, 0) + 1
        total = server_round
        for cid, cnt in self.participation_history.items():
            self.participation_freq[cid] = cnt / total
        # 计算聚合权重并合并参数
        aggregated_params = super().aggregate_fit(server_round, results, failures)
        # 保存检查点
        # save_experiment_state(server_round, ...)
        return aggregated_params
