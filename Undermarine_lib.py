import numpy as np
import cv2
import os

# Transformation matrices
Txyz = np.array([[0.5141, 0.3239, 0.1604],
                 [0.2651, 0.6702, 0.0641],
                 [0.0241, 0.1288, 0.8444]])

Tlms = np.array([[0.3897, 0.6890, 0.0787],
                 [-0.2298, 1.1834, 0.0464],
                 [0.0000, 0.0000, 1.0000]])

Tpca1 = np.array([[1/np.sqrt(3), 0, 0],
                  [0, 1/np.sqrt(6), 0],
                  [0, 0, 1/np.sqrt(2)]])

Tpca2 = np.array([[1, 1, 1],
                  [1, 1, -2],
                  [1, -1, 0]])

Tpca = Tpca1 @ Tpca2


# Helper functions to transform between color spaces
def adjust_gamma(image, gamma=1.0):
    """Applies gamma correction to an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def rgb_to_xyz(im):
    """Converts an RGB image to XYZ."""
    return im @ Txyz.T

def xyz_to_lms(im):
    """Converts an XYZ image to LMS."""
    return im @ Tlms.T

def lms_to_lab(im):
    """Converts an LMS image to lαβ using PCA transformation."""
    return np.log(im + 1e-6) @ Tpca2.T @ Tpca1.T

def lab_to_rgb(imLabCorrec):
    """Converts a corrected lαβ image back to RGB."""
    return np.exp(imLabCorrec @ np.linalg.inv(Tpca.T)) @ np.linalg.inv(Txyz.T @ Tlms.T)

def correct_gray_world(imLab):
    """Applies gray-world correction to the chromatic channels α and β."""
    imLab[:, :, 0] = cv2.equalizeHist(imLab[:, :, 0].astype(np.uint8))
    imLab[:, :, 1] = imLab[:, :, 1] - np.mean(imLab[:, :, 1])  # α correction
    imLab[:, :, 2] = imLab[:, :, 2] - np.mean(imLab[:, :, 2])  # β correction
    return imLab

# Main function to process the image
def process_underwater_image(image_path):
    """Processes an underwater image by correcting color using gray-world assumption."""
    # Load the image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Apply gamma correction
    im_gamma = adjust_gamma(im, gamma=2.4)

    # Normalize the image
    im_norm = im_gamma.astype(np.float32) / 255.0

    # Step 1: RGB to XYZ to LMS to Lab (lαβ)
    imXYZ = rgb_to_xyz(im_norm)
    imLMS = xyz_to_lms(imXYZ)
    imLab = lms_to_lab(imLMS)

    # Step 2: Apply gray-world correction to Lab space
    imLabCorrec = correct_gray_world(imLab)

    # Step 3: Convert corrected Lab (lαβ) back to RGB
    imLabInv = np.clip(adjust_gamma( lab_to_rgb(imLabCorrec), gamma=1/2.4) , -1, 1)

    return imLabInv

def gray_world_correction_BGR(image):
    """
    Apply gray-world correction to an image in BGR space (used by OpenCV).
    The correction normalizes the image's color channels to balance the overall illumination.
    
    :param image: Input image in BGR format
    :return: Corrected image in BGR format
    """
    # Split the image into its B, G, R channels
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    # Calculate the mean of each channel
    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)

    # Calculate the average illuminant
    mean_illuminant = (R_mean + G_mean + B_mean) / 3

    # Apply the gray-world correction on each channel
    R = (R * (mean_illuminant / R_mean)).clip(0, 255)
    G = (G * (mean_illuminant / G_mean)).clip(0, 255)
    B = (B * (mean_illuminant / B_mean)).clip(0, 255)

    # Stack the corrected channels back together
    corrected_image = np.stack([B, G, R], axis=-1).astype(np.uint8)

    return corrected_image

def save_image(image, input_path, output_path):
    """
    Save the corrected image with the prefix 'EX' in the output directory using OpenCV.
    
    :param image: The image to be saved
    :param input_path: The path of the original image
    :param output_path: The directory where the corrected image will be saved
    """
    # Generate a new filename with 'EX_' prefix
    new_image_name = f"EX_{os.path.basename(input_path)}"
    save_path = os.path.join(output_path, new_image_name)
    
    # Save the image using OpenCV's imwrite function
    cv2.imwrite(save_path, image)
    
    print(f"Image ({new_image_name}) saved at {output_path}.")

def delete_images(directory):
    """
    Delete all images in the specified directory that start with the 'EX' prefix.
    
    :param directory: The directory to search for images to delete
    """
    for filename in os.listdir(directory):
        # Check if the file starts with 'EX'
        if filename.startswith('EX'):
            file_path = os.path.join(directory, filename)
            try:
                # Remove the file
                os.remove(file_path)
                print(f"Image deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")