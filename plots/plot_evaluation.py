import os
import json
import matplotlib.pyplot as plt

def plot_metrics(json_file, output_folder):
    with open(json_file, "r") as f:
        data = json.load(f)

    epochs = []
    train_loss = []
    train_map = []
    validation_iou = []
    validation_map = []

    for epoch, metrics in data.items():
        epochs.append(int(epoch.split("_")[1]))  
        train_loss.append(metrics["training_loss"]["train_loss"])
        train_map.append(metrics["training_loss"]["train_map"])
        validation_iou.append(metrics["validation_iou"])
        validation_map.append(metrics["validation_map"])

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_loss, label="Training Loss", marker="o")
    plt.plot(epochs, train_map, label="Training mAP", marker="o")
    plt.plot(epochs, validation_iou, label="Validation IoU", marker="o")
    plt.plot(epochs, validation_map, label="Validation mAP", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"Metrics Over Epochs - {os.path.basename(json_file)}")
    plt.legend()
    plt.grid(True)


    output_path = os.path.join(output_folder, f"{os.path.basename(json_file).replace('.json', '.png')}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved: {output_path}")


def process_json_files(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            json_file = os.path.join(input_folder, file_name)
            plot_metrics(json_file, output_folder)



if __name__ == "__main__":
    #plot every json file in the evaluation_results folder
    # and save the plots in the plots folder
    input_folder = "evaluation_results"  
    output_folder = "plots"  
    process_json_files(input_folder, output_folder)
