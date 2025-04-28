import matplotlib.pyplot as plt
import numpy as np

def plot_performance_by_resource_group(results, resource_groups):
    """绘制不同资源组的性能分布箱线图"""
    # results: dict client_id->accuracy, assume mapping client->group
    groups = resource_groups.keys()
    data = {g: [] for g in groups}
    for cid, acc in results.items():
        grp = results[cid]["group"]
        data[grp].append(acc)
    plt.boxplot([data[g] for g in groups], labels=groups)
    plt.ylabel("Accuracy")
    plt.title("Performance by Resource Group")
    plt.show()


def plot_convergence_curves(history, target_accuracy=0.8):
    rounds = list(range(1, len(history)+1))
    plt.plot(rounds, history, label="Global Accuracy")
    idx = np.argmax(np.array(history) >= target_accuracy)
    if history[idx] >= target_accuracy:
        plt.scatter(idx+1, history[idx], color='red',
                    label=f"Hit {target_accuracy}@{idx+1}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def calculate_resource_efficiency(accuracy_gain, resource_consumption):
    return accuracy_gain / resource_consumption

