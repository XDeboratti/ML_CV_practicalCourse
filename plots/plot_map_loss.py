import os
import json
import matplotlib.pyplot as plt

def load_metrics(json_file, max_epochs=100):
    with open(json_file, "r") as f:
        data = json.load(f)

    epochs = []
    train_map = []
    validation_map = []

    for epoch, metrics in data.items():
        epoch_num = int(epoch.split("_")[1])
        if epoch_num <= max_epochs:
            epochs.append(epoch_num)
            train_map.append(metrics["training_loss"]["train_map"])
            validation_map.append(metrics["validation_map"])

    return epochs, train_map, validation_map

def plot_metrics(json_file, model_name, max_epochs=25):
    epochs, train_map, validation_map = load_metrics(json_file, max_epochs=max_epochs)
    
    plt.figure(figsize=(12, 8))
    
    # mAP Plot
    plt.plot(epochs, train_map, label="Train mAP", marker="o")
    plt.plot(epochs, validation_map, label="Validation mAP", marker="o")
    
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.title(f"Train and Validation mAP - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    json_file = "evaluation_results/metrics_DFG_NO_AUG_3_LAYERS_LONG_RUN.json"
    model_name = "Faster R-CNN with Augmentation"
    
    plot_metrics(json_file, model_name, max_epochs=100)
