import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Transformation matrices
Txyz = np.array([[0.5141, 0.3239, 0.1604],
                 [0.2651, 0.6702, 0.0641],
                 [0.0241, 0.1288, 0.8444]])

Tlms = np.array([[0.3897, 0.6890, -0.0787],
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
    """Apply gamma correction to an image."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def rgb_to_xyz(im):
    """Convert an RGB image to XYZ color space."""
    return im @ Txyz.T

def xyz_to_lms(im):
    """Convert an XYZ image to LMS color space."""
    return im @ Tlms.T

def lms_to_lab(im):
    """Convert an LMS image to Lab color space using PCA transformation."""
    return np.log(im + 1e-6) @ Tpca2.T @ Tpca1.T

def lab_to_rgb(imLabCorrec):
    """Convert a corrected Lab image back to RGB."""
    return np.exp(imLabCorrec @ np.linalg.inv(Tpca.T)) @ np.linalg.inv(Txyz.T @ Tlms.T)

def correct_gray_world(imLab):
    """Apply gray-world correction to the chromatic channels α and β."""
    # imLab[:, :, 0] = histogram_equal(imLab[:, :, 0])
    imLab[:, :, 1] -= np.mean(imLab[:, :, 1])  # α correction
    imLab[:, :, 2] -= np.mean(imLab[:, :, 2])  # β correction

# Main function to process the image
def process_underwater_image(image_path):
    """Process an underwater image using gray-world correction."""
    # Load the image
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Apply gamma correction
    im_gamma = adjust_gamma(im, gamma=1)

    # Normalize the image
    im_norm = im_gamma.astype(np.float32) / 255.0

    # Step 1: RGB to XYZ to LMS to Lab
    imXYZ = rgb_to_xyz(im_norm)
    imLMS = xyz_to_lms(imXYZ)
    imLab = lms_to_lab(imLMS)

    # display_Lab(imLab)

    # Step 2: Apply gray-world correction in Lab space
    correct_gray_world(imLab)

    # display_Lab(imLab)

    # Step 3: Convert corrected Lab back to RGB
    imLabInv = (np.clip(lab_to_rgb(imLab), 0, 1) * 255).astype(np.uint8)

    return cv2.cvtColor(imLabInv, cv2.COLOR_RGB2BGR)

def histogram_equal(channel):
    """Equalize histogram of a channel after converting from log space."""
    non_log_channel = np.clip(np.exp(channel), 0, 1)
    non_log_channel = (non_log_channel * 255).astype(np.uint8)
    non_log_equalized_channel = cv2.equalizeHist(non_log_channel)
    non_log_equalized_channel = non_log_equalized_channel.astype(np.float32) / 255.0
    equalized_channel = np.log(non_log_equalized_channel + 1e-06)

    return equalized_channel

def gray_world_correction_BGR(image):
    """Apply gray-world correction to an image in BGR format."""
    B, G, R = cv2.split(image)
    R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
    mean_illuminant = (R_mean + G_mean + B_mean) / 3
    R = (R * (mean_illuminant / R_mean)).clip(0, 255)
    G = (G * (mean_illuminant / G_mean)).clip(0, 255)
    B = (B * (mean_illuminant / B_mean)).clip(0, 255)
    corrected_image = np.stack([B, G, R], axis=-1).astype(np.uint8)

    return corrected_image

def white_patches_correction_BGR(image):
    """Apply white patches correction to an image in BGR format."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_patch_pixels = int(0.02 * gray_image.size)
    indices = np.unravel_index(np.argsort(gray_image, axis=None)[-num_patch_pixels:], gray_image.shape)
    bright_patches = image[indices]
    mean_patch = np.mean(bright_patches, axis=0)
    white_point = np.array([255, 255, 255])
    correction_factor = white_point / mean_patch
    corrected_image = np.clip(image * correction_factor, 0, 255).astype(np.uint8)

    return corrected_image

def save_image(image, input_path, output_path):
    """Save the corrected image with the prefix 'EX' in the output directory."""
    new_image_name = f"EX_{os.path.basename(input_path)}"
    save_path = os.path.join(output_path, new_image_name)
    cv2.imwrite(save_path, image)
    print(f"Image ({new_image_name}) saved at {output_path}.")


def delete_images(directory):
    """Delete images in the specified directory that start with 'EX'."""
    for filename in os.listdir(directory):
        if filename.startswith('EX'):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Image deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")


def display_Lab(lab_image):
    """Display the L, α, and β channels."""
    l = lab_image[:, :, 0].flatten()      # L channel
    alpha = lab_image[:, :, 1].flatten()  # α channel
    beta = lab_image[:, :, 2].flatten()   # β channel

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot of α vs β
    ax[0].scatter(alpha, beta, s=1, color='blue', alpha=0.5)
    ax[0].set_title('αβ Diagram')
    ax[0].set_xlabel('α (green-red component)')
    ax[0].set_ylabel('β (blue-yellow component)')
    ax[0].set_xlim(np.min(alpha), np.max(alpha))
    ax[0].set_ylim(np.min(beta), np.max(beta))
    ax[0].grid(True)

    # Histogram of the L channel
    ax[1].hist(l, bins=50, color='gray', alpha=0.7)
    ax[1].set_title('Luminance (L) Histogram')
    ax[1].set_xlabel('Luminance Value')
    ax[1].set_ylabel('Frequency')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
