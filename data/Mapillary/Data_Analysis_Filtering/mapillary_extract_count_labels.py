import os
import json

def count_label_instances(annotation_dir, output_file="label_counts.json"):
    """
    Scans all JSON annotation files in a directory, counts the frequency of each label,
    and saves the result in a JSON file.
    """
    label_counts = {}

    #browse through all json files in the directory
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith(".json"):
            file_path = os.path.join(annotation_dir, annotation_file)
            with open(file_path, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)

                #counts labels
                for obj in annotation_data.get("objects", []):
                    label = obj.get("label", "unknown")
                    label_counts[label] = label_counts.get(label, 0) + 1

    #saves results in json file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(label_counts, f, indent=4, ensure_ascii=False)

    print(f"Label-intances counted and saved in {output_file}.")
    print(f"Found Labels: {len(label_counts)}")

#directory with annotations
annotation_dir = "train/annotations"

#counts labels
count_label_instances(annotation_dir)
