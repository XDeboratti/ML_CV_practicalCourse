import os
from collections import Counter
from PIL import Image

def get_resolution_distribution(image_dir):
    """
    Computes the frequency distribution of image resolutions in a folder.

    :param image_dir: Path to the image directory.
    :return: Counter object with (width, height) as keys and counts as values.
    """
    resolution_counter = Counter()

    for file_name in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file_name)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    resolution_counter[(width, height)] += 1
            except Exception as e:
                print(f"Skipping file {file_name}: {e}")
    
    return resolution_counter

if __name__ == "__main__":
    image_dir = "images_relevant"

    if not os.path.isdir(image_dir):
        print(f"Directory '{image_dir}' does not exist.")
    else:
        resolution_counts = get_resolution_distribution(image_dir)

        print("\nTop 10 most common resolutions:")
        for res, count in resolution_counts.most_common(10):
            print(f"{res[0]}x{res[1]}: {count} images")

        total_images = sum(resolution_counts.values())
        unique_resolutions = len(resolution_counts)
        print(f"\nTotal images processed: {total_images}")
        print(f"Unique resolutions found: {unique_resolutions}")
