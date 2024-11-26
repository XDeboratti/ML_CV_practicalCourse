import os
import shutil
import random

#die splits müssen zusamen 1 ergeben, die ratios train_ration + val_ratio + test_ratio
#splittet den Datensatz (Bilder und Annotations) in train test und val auf jeweils in dem ratio wie angegeben
#dafür mischt er die Reihenfolge und selectiert dann je nach ratio und füllt diese in die entsprechenden ordner:
#mtsd_v2_fully_annotated/train
#       -images
#       -annotations
#       /test
#       -images
#       -annotations
#       /val
#       -images
#       -annotations

def split_dataset(images_dir, annotations_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    

    #Sammle alle Dateien mit deren Namen: ["image1.jpg", "image2.png",..]
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])

    #mischt die Daten paarweise   
    data_pairs = list(zip(image_files, annotation_files))
    random.shuffle(data_pairs) 

    #aufteilen der Daten je nach ratio
    num_total = len(data_pairs)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)

    train_data = data_pairs[:num_train]
    val_data = data_pairs[num_train:num_train + num_val]
    test_data = data_pairs[num_train + num_val:]

    #kopiert die Daten in die Ordner
    def copy_files(data, split):
        for image_file, annotation_file in data:
            shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, split, 'images', image_file))
            shutil.copy(os.path.join(annotations_dir, annotation_file), os.path.join(output_dir, split, 'annotations', annotation_file))

    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    copy_files(test_data, 'test')

    print(f"Dataset erfolgreich aufgeteilt: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")


images_dir = "images_relevant"
annotations_dir = "annotations_relevant"
output_dir = "."
split_dataset(images_dir, annotations_dir, output_dir)
