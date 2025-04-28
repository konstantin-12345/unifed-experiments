# File: unifed_client.py
import argparse
import flwr as fl
from config.experiment_config import ExperimentConfig
from utils.resource_monitor import ResourceMonitor
from utils.utils import compute_rci, compute_dfs

class UniFedClient(fl.client.NumPyClient):
    def __init__(self, cid, train, test, cfg):
        self.cid = cid
        self.train = train
        self.test = test
        self.cfg = cfg
        # 初始化模型
        # self.model = create_model()

    def get_parameters(self):
        # 从模型中提取参数
        return []

    def fit(self, parameters, config):
        # 更新模型
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        # 执行本地训练轮次
        # TODO: 根据 RCI 计算 Ei
        metrics = monitor.get_measurements()
        # 计算 RCI, DFS
        # TODO: compute_rci, compute_dfs
        return parameters, len(self.train), {"client_id": self.cid, **metrics}

    def evaluate(self, parameters, config):
        # 本地评估
        loss, acc = 0.0, 0.0
        return loss, len(self.test), {"accuracy": acc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--data", type=str)
    parser.add_argument("--config-path", type=str, default="clients_config.json")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    # TODO: 加载并切分数据
    train, test = None, None

    client = UniFedClient(args.client_id, train, test, cfg)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
