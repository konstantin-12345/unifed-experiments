# unifed_server.py
import argparse
import flwr as fl
from strategy import UniFedStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    strategy = UniFedStrategy(
        fraction_fit=0.1,  # 选取比例
        min_fit_clients=10,
        min_available_clients=10,
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
