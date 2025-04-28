import os
import torch
import json

def save_experiment_state(round_num, global_model, metrics, client_states, path):
    os.makedirs(path, exist_ok=True)
    torch.save(global_model.state_dict(),
               os.path.join(path, f"global_model_round_{round_num}.pt"))
    with open(os.path.join(path, f"metrics_round_{round_num}.json"), 'w') as f:
        json.dump(metrics, f)
    with open(os.path.join(path, f"client_states_round_{round_num}.json"), 'w') as f:
        json.dump(client_states, f)


def load_experiment_state(round_num, model_class, path):
    model = model_class()
    state = torch.load(os.path.join(path, f"global_model_round_{round_num}.pt"))
    model.load_state_dict(state)
    with open(os.path.join(path, f"metrics_round_{round_num}.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(path, f"client_states_round_{round_num}.json")) as f:
        client_states = json.load(f)
    return model, metrics, client_states

