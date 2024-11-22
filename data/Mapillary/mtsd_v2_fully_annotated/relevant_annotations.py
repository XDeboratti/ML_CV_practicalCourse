#picks relevant annotations (look below target labels) out of annotations 
import os
import json

def filter_annotations(annotation_dir, target_labels, output_annotation_dir):

    os.makedirs(output_annotation_dir, exist_ok=True)
    relevant_count = 0 

    for filename in os.listdir(annotation_dir):
        if filename.endswith('.json'):
            annotation_path = os.path.join(annotation_dir, filename)
            with open(annotation_path, 'r') as fid:
                annotation = json.load(fid)

            relevant_objects = [
                obj for obj in annotation['objects'] if obj['label'] in target_labels
            ]

            if relevant_objects:
                relevant_count += 1
                annotation['objects'] = relevant_objects  
                output_annotation_path = os.path.join(output_annotation_dir, filename)
                with open(output_annotation_path, 'w') as fid:
                    json.dump(annotation, fid, indent=4)

    print(f"Gefilterte Annotationen: {relevant_count} Dateien mit relevanten Labels gespeichert.")

if __name__ == "__main__":
    annotation_dir = "annotations"  
    output_annotation_dir = "annotations_relevant"  

    target_labels = ["regulatory--stop--g1","regulatory--yield--g1","regulatory--priority-road--g4"]

    filter_annotations(annotation_dir, target_labels, output_annotation_dir)
