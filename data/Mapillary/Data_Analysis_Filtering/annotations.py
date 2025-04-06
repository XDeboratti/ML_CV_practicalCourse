# This script analyzes the annotations in a directory.
# It lists all labels present in the dataset (e.g., traffic signs)
# and counts how often each label appears in the annotations.
# At the end, it outputs the total number of annotated objects.
import os
import json
from collections import Counter

def extract_labels(annotation_dir):
    labels = []
    
    
    for filename in os.listdir(annotation_dir):
        if filename.endswith('.json'):
            with open(os.path.join(annotation_dir, filename), 'r') as fid:
                annotation = json.load(fid)
                
                for obj in annotation.get('objects', []):
                    labels.append(obj['label'])
    
    label_counts = Counter(labels)
    
    return label_counts

if __name__ == "__main__":
    annotation_dir = "annotations_relevant" 
    label_counts = extract_labels(annotation_dir)
    sum = 0
    print("Gefundene Labels und ihre Häufigkeit:")
    for label, count in label_counts.items():
        sum += count
        print(f"{label}: {count}")
    print(sum)