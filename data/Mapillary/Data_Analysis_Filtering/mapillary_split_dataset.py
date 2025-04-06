import os
import shutil
import random
# The splits must add up to 1: train_ratio + val_ratio + test_ratio
# This function splits a dataset (images and annotations) into train, val, and test sets
# according to the specified ratios. It shuffles the data, selects subsets, and copies the files
# into the following folder structure under 'output_dir':
#/mtsd_v2_fully_annotated
#       /train
#       -images
#       -annotations
#       /test
#       -images
#       -annotations
#       /val
#       -images
#       -annotations

#one call just for quick test dataset
def split_dataset(images_dir, annotations_dir, output_dir, train_ratio=0.05, val_ratio=0.02, test_ratio=0.03):
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])

    #pair images with their corresponding annotations and shuffle the order   
    data_pairs = list(zip(image_files, annotation_files))
    random.shuffle(data_pairs) 

    #split data according to the specified ratios
    num_total = len(data_pairs)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)

    train_data = data_pairs[:num_train]
    val_data = data_pairs[num_train:num_train + num_val]
    test_data = data_pairs[num_train + num_val:]

    #helper function to copy files into the appropriate split folders
    def copy_files(data, split):
        for image_file, annotation_file in data:
            shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, split, 'images', image_file))
            shutil.copy(os.path.join(annotations_dir, annotation_file), os.path.join(output_dir, split, 'annotations', annotation_file))

    copy_files(train_data, 'train_test')
    copy_files(val_data, 'val_test')
    copy_files(test_data, 'test_test')

    print(f"Dataset succesfull splittet: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")


images_dir = "images_relevant"
annotations_dir = "annotations_relevant"
output_dir = "."
split_dataset(images_dir, annotations_dir, output_dir)
