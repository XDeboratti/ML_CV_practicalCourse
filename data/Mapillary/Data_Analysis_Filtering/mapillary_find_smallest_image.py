import os
from PIL import Image

def find_smallest_image(directory ="train/images"):
    """
    Find the image with the smallest dimensions in a directory.

    :param directory: Path to the directory containing images.
    :return: A tuple containing (file_name, width, height).
    """
    smallest_image = None
    smallest_dimensions = (float('inf'), float('inf'))  # Start with very large dimensions

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if (width * height) < (smallest_dimensions[0] * smallest_dimensions[1]):
                    smallest_dimensions = (width, height)
                    smallest_image = file_name
        except Exception as e:
            print(f"Could not process file {file_name}: {e}")

    if smallest_image:
        return smallest_image, smallest_dimensions[0], smallest_dimensions[1]
    else:
        return None, None, None

if __name__ == "__main__":
    directory = input("Enter the path to the image folder: ").strip()

    if not os.path.isdir(directory):
        print("The specified path is not a directory.")
    else:
        smallest_image, width, height = find_smallest_image(directory)
        if smallest_image:
            print(f"Smallest image: {smallest_image}")
            print(f"Dimensions: {width}x{height} pixels")
        else:
            print("No valid images found in the directory.")