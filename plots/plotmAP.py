import os
import json
import matplotlib.pyplot as plt

def load_map_values(json_file, max_epochs=100):
    with open(json_file, "r") as f:
        data = json.load(f)

    epochs = []
    validation_map = []

    for epoch, metrics in data.items():
        epoch_num = int(epoch.split("_")[1])
        if epoch_num <= max_epochs:
            epochs.append(epoch_num)
            validation_map.append(metrics["validation_map"])

    return epochs, validation_map

def plot_map_comparison(json_files, model_names, max_epochs=25):
    plt.figure(figsize=(12, 8))

    for json_file, model_name, num in zip(json_files, model_names, ["52", "34"]):
        epochs, validation_map = load_map_values(json_file, max_epochs=max_epochs)
        highest_map = max(validation_map) if validation_map else 0
        model_label = f"{model_name}, mAP@50 = {num}%"
        plt.plot(epochs, validation_map, label=model_label, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.title("Validation mAP Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    json_files = [
        "evaluation_results/metrics_DFG_NO_AUG_5_LAYERS_COSINE_ANNEALING_LONG_RUN_1.json",
        "evaluation_results/metrics_DFG_NO_AUG_3_LAYERS_LONG_RUN.json",
    ]
    
    model_names = [
        "high start Learning Rate & 5 BBone Layers",
        "low start Learning Rate & 3 BBone Layers",
    ]
    
    plot_map_comparison(json_files, model_names, max_epochs=100)