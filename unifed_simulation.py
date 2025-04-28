# unifed_simulation.py
import argparse
import flwr as fl
from strategy import UniFedStrategy
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

def get_data(dataset_name, num_clients):
    # 下载并划分到 num_clients 份，返回 dict client_id→(train, test)
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--simulate", type=bool, default=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    # 1. 加载 & 划分数据
    client_data = get_data(args.data, args.clients)

    # 2. 定义客户端工厂
    def client_fn(cid):
        # 返回实现了 get_parameters, fit, evaluate 的 Flower client，
        # client 内部要测量 C, M, E, B 并上报 RCI/DFS
        pass

    # 3. 启动模拟
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=UniFedStrategy(),
    )

if __name__ == "__main__":
    main()
