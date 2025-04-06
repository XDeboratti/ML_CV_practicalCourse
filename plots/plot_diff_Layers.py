import os
import json
import matplotlib.pyplot as plt

def load_map_values(json_file, max_epochs=12):
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

def plot_map_comparison(json_files, model_names, max_epochs=12):
    plt.figure(figsize=(12, 8))
    colors = [("darkblue", "blue"), ("darkgreen", "green"), ("darkred", "red")]
    
    for i, (json_file, model_name) in enumerate(zip(json_files, model_names)):
        epochs, train_map, validation_map = load_map_values(json_file, max_epochs=max_epochs)
        dark_color, light_color = colors[i]
        
        plt.plot(epochs, train_map, label=f"{model_name} - Train mAP", color=dark_color, marker="o")
        plt.plot(epochs, validation_map, label=f"{model_name} - Validation mAP", color=light_color, marker="o")
    
    plt.xlabel("Epoch")
    plt.ylabel("mAP@50")
    plt.title("Train and Validation mAP Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    json_files = [
        "evaluation_results/metrics_DFG_NO_AUG_1_LAYERS.json",
        "evaluation_results/metrics_DFG_NO_AUG_3_LAYERS.json",
        "evaluation_results/metrics_DFG_NO_AUG_5_LAYERS_COSINE_ANNEALING_LONG_RUN.json"
    ]
    
    model_names = [
        "Faster R-CNN with 1 trainable Backbone Layer",
        "Faster R-CNN with 3 trainable Backbone Layers",
        "Faster R-CNN with 5 trainable Backbone Layers"
    ]
    
    plot_map_comparison(json_files, model_names, max_epochs=12)
