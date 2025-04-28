# unifed_client.py
import argparse
import flwr as fl
from model import create_model  # 你自己的模型定义
from utils import compute_rci, compute_dfs

class UniFedClient(fl.client.NumPyClient):
    def __init__(self, cid, train_data, test_data, config):
        # 保存 cid, 数据, config（RCI 初值、超参等）
        pass

    def get_parameters(self):
        # 返回当前模型参数
        pass

    def fit(self, parameters, config):
        # 1. 更新模型到 parameters
        # 2. 测量 C, M, E, B → RCI, DFS
        # 3. 本地训练 E_i 轮
        # 4. 计算软标签 p_i, Δ_i 上报 metrics 和 model_diff
        return model_parameters, len(train_data), {"RCI": rci, "soft_labels": p_i, ...}

    def evaluate(self, parameters, config):
        # 本地测试，返回 loss, metric, e.g. accuracy
        return loss, len(test_data), {"accuracy": acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--data", type=str)
    parser.add_argument("--config-path", type=str, default="clients_config.json")
    args = parser.parse_args()

    # 1. 读取 clients_config.json 拿到本客户端初始资源
    # 2. 加载对应数据切片 train/test
    # 3. create UniFedClient 并启动
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=UniFedClient(...),
    )

if __name__ == "__main__":
    main()
