#filters accordingly to the annotations the right images.
import os
import json
import shutil

def filter_images(annotation_dir, image_dir, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    relevant_count = 0

    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.json'):
            annotation_path = os.path.join(annotation_dir, annotation_file)
            with open(annotation_path, 'r') as fid:
                annotation = json.load(fid)
            
            image_key = os.path.splitext(annotation_file)[0]
            image_filename = f"{image_key}.jpg"  
            
            image_path = os.path.join(image_dir, image_filename)
            if os.path.exists(image_path):
                relevant_count += 1

                shutil.copy(image_path, os.path.join(output_image_dir, image_filename))

    print(f"Gefilterte Bilder: {relevant_count} relevante Bilder gespeichert.")


if __name__ == "__main__":
    annotation_dir = "/graphics/scratch2/students/kornwolfd/ML_CV_practicalCourse/data_RoadSigns/mapillary/annotations_relevant"  
    image_dir = "/graphics/scratch2/students/kornwolfd/ML_CV_practicalCourse/data_RoadSigns/mapillary/images" 
    output_image_dir = "/graphics/scratch2/students/kornwolfd/ML_CV_practicalCourse/data_RoadSigns/mapillary/images_relevant" 

    filter_images(annotation_dir, image_dir, output_image_dir)
