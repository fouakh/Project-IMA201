import argparse
import os
import cv2
from Undermarine_lib import *

def main(saving, directory):
    """Process or delete images in the specified directory (not 'Images')."""

    if directory == 'Images':
        raise ValueError("The directory cannot be 'Images' ")

    if not os.path.exists(directory):
        os.makedirs(directory)

    if saving == "off":
        delete_images(directory)
    elif saving == "on":
        for filename in os.listdir("Images"):
            file_path = os.path.join("Images", filename)

            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                im = cv2.imread(file_path)

                if im is not None:
                    imc = process_underwater_image(im, "BGR", "GW")
                    # imc = ace_enhance_image_poly(im, 7, 11, '1/r')
                    save_image(imc, filename, directory)
                else:
                    print(f"Error: Unable to load image {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process or delete images in a specified directory.")
    parser.add_argument('-s', '--saving', type=str, required=True, help='Enable image correction and saving')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory for processing or deleting images (cannot be "Images")')
    args = parser.parse_args()
    main(saving=args.saving, directory=args.directory)
