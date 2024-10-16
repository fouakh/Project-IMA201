import os
import cv2
import argparse
from Undermarine_lib import *

def main(on):
    if not on:
        # Delete all images with prefix 'EX' in the 'Images' directory
        delete_images('Images')
    else:
        # Process and correct images in the 'Images' directory
        for filename in os.listdir('Images'):
            file_path = os.path.join('Images', filename)

            # Check for valid image extensions
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                # Load the image using OpenCV
                im = cv2.imread(file_path)

                if im is not None:
                    # Apply the underwater correction (gray-world correction)
                    imc = gray_world_correction_BGR(im)

                    # Save the corrected image with 'EX_' prefix
                    save_image(imc, filename, 'Images')
                else:
                    print(f"Error: Unable to load image {filename}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process or delete images in the 'Images' directory")
    parser.add_argument('--on', action='store_true', help='Turn ON image correction and saving')
    args = parser.parse_args()
    main(on=args.on)

