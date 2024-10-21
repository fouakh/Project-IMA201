import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

########################################################################################
# BGR Correction
########################################################################################

def gray_world_BGR(image):
    # Apply gray-world correction to an image in BGR format.
    B, G, R = cv2.split(image)  
    R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B) 
    mean_illuminant = (R_mean + G_mean + B_mean) / 3  

    R = (R * (mean_illuminant / R_mean)).clip(0, 255)
    G = (G * (mean_illuminant / G_mean)).clip(0, 255)
    B = (B * (mean_illuminant / B_mean)).clip(0, 255)
    corrected_image = np.stack([B, G, R], axis=-1).astype(np.uint8)  

    return corrected_image


def white_patches_BGR(image):
    # Apply white patches correction to an image in BGR format.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    num_patch_pixels = int(0.02 * gray_image.size) 
    indices = np.unravel_index(np.argsort(gray_image, axis=None)[-num_patch_pixels:], gray_image.shape)
    bright_patches = image[indices] 
    mean_patch = np.mean(bright_patches, axis=0) 
    white_point = np.array([255, 255, 255]) 
    correction_factor = white_point / mean_patch 
    corrected_image = np.clip(image * correction_factor, 0, 255).astype(np.uint8)  

    return corrected_image


########################################################################################
# Lab Correction
########################################################################################

# Transformation matrices.
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


def adjust_gamma(image, gamma=1.0):
    # Apply gamma correction to an image.
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def rgb_to_xyz(im):
    # Convert an RGB image to XYZ color space.
    return im @ Txyz.T  


def xyz_to_lms(im):
    # Convert an XYZ image to LMS color space.
    return im @ Tlms.T 


def lms_to_lab(im):
    # Convert an LMS image to Lab color space using PCA transformation.
    return np.log(im + 1e-6) @ Tpca2.T @ Tpca1.T 


def lab_to_rgb(imLabCorrec):
    # Convert a corrected Lab image back to RGB.
    return np.exp(imLabCorrec @ np.linalg.inv(Tpca.T)) @ np.linalg.inv(Txyz.T @ Tlms.T)  


def gray_world_lab(imLab):
    # Apply gray-world correction to the chromatic channels a and b
    imLab[:, :, 1] -= np.mean(imLab[:, :, 1])  
    imLab[:, :, 2] -= np.mean(imLab[:, :, 2])  


def white_patches_lab(imLab, percent=2.0):
    # Apply white-patches correction to the a and b channels based on the brightest specified percentage of pixels.
    num_pixels = imLab[:, :, 0].size  
    num_brightest_pixels = int(num_pixels * percent / 100)  
    bright_index = np.argsort(imLab[:, :, 0], axis=None)[-num_brightest_pixels:]  

    mean_a = np.mean(imLab[:, :, 1].flatten()[bright_index])
    mean_b = np.mean(imLab[:, :, 2].flatten()[bright_index])

    imLab[:, :, 1] -= mean_a  
    imLab[:, :, 2] -= mean_b  


def spacial_white_patches_lab(imLab, kernel_size=5):
    # Apply white-patches correction to the a and b channels using a local mean filter.
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2) 
    filtered_L = cv2.filter2D(imLab[:, :, 0], -1, kernel)  
    max_idx = np.unravel_index(np.argmax(filtered_L), filtered_L.shape)  
    
    half_kernel = kernel_size // 2
    y_min = max(max_idx[0] - half_kernel, 0)
    y_max = min(max_idx[0] + half_kernel + 1, imLab.shape[0])
    x_min = max(max_idx[1] - half_kernel, 0)
    x_max = min(max_idx[1] + half_kernel + 1, imLab.shape[1])
    
    alpha_region = imLab[y_min:y_max, x_min:x_max, 1] 
    beta_region = imLab[y_min:y_max, x_min:x_max, 2]
    
    alpha_mean = np.mean(alpha_region)
    beta_mean = np.mean(beta_region)

    imLab[:, :, 1] -= alpha_mean 
    imLab[:, :, 2] -= beta_mean


########################################################################################
# Main function for Correction
########################################################################################

def process_underwater_image(imBGR, color_space="Lab", hyp="GW"):
    # Process an underwater image using color correction methods. 
    if color_space == "Lab":
        im = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB) 
        im_gamma = adjust_gamma(im, gamma=1)  
        im_norm = im_gamma.astype(np.float32) / 255.0 

        imXYZ = rgb_to_xyz(im_norm)
        imLMS = xyz_to_lms(imXYZ)
        imLab = lms_to_lab(imLMS)  

        if hyp == "GW":
            gray_world_lab(imLab) 
        elif hyp == "WP":
            white_patches_lab(imLab)
        elif hyp == "SWP":
            spacial_white_patches_lab(imLab)

        imLabInv = (np.clip(lab_to_rgb(imLab), 0, 1) * 255).astype(np.uint8)

        return cv2.cvtColor(imLabInv, cv2.COLOR_RGB2BGR)
    
    elif color_space == "BGR":
        imc = imBGR.copy()
        if hyp == "GW":
            imc = gray_world_BGR(imc)
        elif hyp == "WP":
            imc = white_patches_BGR(imc)

        return imc


def histogram_equal(channel):
    # Equalize histogram of a channel after converting from log space.
    non_log_channel = np.clip(np.exp(channel), 0, 1)  
    non_log_channel = (non_log_channel * 255).astype(np.uint8)  
    non_log_equalized_channel = cv2.equalizeHist(non_log_channel)  
    non_log_equalized_channel = non_log_equalized_channel.astype(np.float32) / 255.0 
    equalized_channel = np.log(non_log_equalized_channel + 1e-06)  

    return equalized_channel


########################################################################################
# Fast Implementation of ACE Algorithm
########################################################################################

def ace_algorithm(image):
    # Apply the ACE algorithm for color correction."""
    pass


########################################################################################
# Utility Functions
########################################################################################

def save_image(image, input_path, output_path):
    # Save the corrected image with the prefix 'EX' in the output directory."""
    new_image_name = f"EX_{os.path.basename(input_path)}"  
    save_path = os.path.join(output_path, new_image_name)  
    cv2.imwrite(save_path, image)  
    print(f"Image ({new_image_name}) saved at {output_path}.")


def delete_images(directory):
    # Delete images in the specified directory that start with 'EX'."""
    for filename in os.listdir(directory):
        if filename.startswith('EX'):  
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)  
                print(f"Image deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")


def display_lab(lab_image):
    # Display the L, a, and b channels of a Lab image."""
    l = lab_image[:, :, 0].flatten()  
    alpha = lab_image[:, :, 1].flatten()  
    beta = lab_image[:, :, 2].flatten()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(alpha, beta, s=1, color='blue', alpha=0.5)  
    ax[0].set_title('αβ Diagram')
    ax[0].set_xlabel('α (green-red component)')
    ax[0].set_ylabel('β (blue-yellow component)')
    ax[0].set_xlim(np.min(alpha), np.max(alpha))
    ax[0].set_ylim(np.min(beta), np.max(beta))
    ax[0].grid(True)

    ax[1].hist(l, bins=50, color='gray', alpha=0.7)  
    ax[1].set_title('Luminance (L) Histogram')
    ax[1].set_xlabel('Luminance Value')
    ax[1].set_ylabel('Frequency')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
