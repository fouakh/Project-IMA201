import argparse
import os
import cv2
from Undermarine_lib import *

def main(on, directory):
    """Process or delete images in the specified directory (not 'Images')."""

    if directory == 'Images':
        raise ValueError("The directory cannot be 'Images' ")

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not on:
        delete_images(directory)
    else:
        for filename in os.listdir("Images"):
            file_path = os.path.join("Images", filename)

            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                im = cv2.imread(file_path)

                if im is not None:
                    # imc = process_underwater_image(im, "BGR", "GW")
                    imc = ace_algorithm(im)
                    save_image(imc, filename, directory)
                else:
                    print(f"Error: Unable to load image {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process or delete images in a specified directory.")
    parser.add_argument('--on', action='store_true', help='Enable image correction and saving')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory for processing or deleting images (cannot be "Images")')
    args = parser.parse_args()
    main(on=args.on, directory=args.directory)
