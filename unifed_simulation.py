# File: unifed_simulation.py
import argparse
import flwr as fl
from strategy import UniFedStrategy
from config.experiment_config import ExperimentConfig

def get_data(dataset_name, num_clients):
    # TODO: 下载并按客户端划分
    raise NotImplementedError

class SimulationClient(fl.client.NumPyClient):
    def __init__(self, cid, train, test):
        self.cid = cid
        self.train = train
        self.test = test
        # 初始化模型、监控器等

    def get_parameters(self):
        # 返回模型参数
        pass

    def fit(self, parameters, config):
        # 本地训练逻辑，计算 RCI/DFS, 软标签, 返回 model_diff 和 metrics
        pass

    def evaluate(self, parameters, config):
        # 返回 loss, num_examples, metrics
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--simulate", type=bool, default=True)
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    client_data = get_data(args.data, args.clients)
    def client_fn(cid):
        train, test = client_data[int(cid)]
        return SimulationClient(cid, train, test)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=UniFedStrategy(),
    )

