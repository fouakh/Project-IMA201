import os
import numpy as np
import cv2
from scipy.special import comb 
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
# Fast Implementation of ACE Algorithm Using Polynomes
########################################################################################

def get_slope_coeffs(alpha, degree):
    if degree == 3:
        slope_coeff_3 = [
            # Coefficients for different alpha values, from 1 to 8
            [1.00000000, 0.00000000], 
            [1.73198742, -0.69610145],
            [2.13277255, -1.21872717],
            [2.38176185, -1.52677041],
            [2.59620299, -1.79723305],
            [2.78241030, -2.03475483],
            [2.94527479, -2.24401889],
            [3.08812041, -2.42848207],
            [3.21463262, -2.59244566],
            [3.32706695, -2.73856141],
            [3.42747864, -2.86932869],
            [3.51818929, -2.98766105],
            [3.60000335, -3.09453402],
            [3.67436451, -3.19178134],
            [3.74278641, -3.28134699],
        ]
        index_alpha = int(2*alpha + 0.5) - 2
        return slope_coeff_3[index_alpha]

    elif degree == 5:
        slope_coeff_5 = [
            [1.00000000, 0.00000000, -0.00000000],
            [1.66723389, -0.36136104, -0.34712113],
            [2.35346531, -2.02651073, 0.61570501],
            [2.94915439, -4.05195249, 2.10249240],
            [3.29516608, -5.31192181, 3.10222415],
            [3.56410808, -6.24932127, 3.80525524],
            [3.80968824, -7.12768947, 4.47258157],
            [4.03354356, -7.94288855, 5.09736922],
            [4.23878049, -8.70009742, 5.68134357],
            [4.42702610, -9.40144877, 6.22474375],
            [4.60005346, -10.05098474, 6.72977516],
            [4.76014304, -10.65555033, 7.20114304],
            [4.90798649, -11.21656584, 7.63952738],
            [5.04473456, -11.73752961, 8.04735155],
            [5.17324675, -12.22873575, 8.43246007]
        ]
        index_alpha = int(2*alpha + 0.5) - 2
        return slope_coeff_5[index_alpha]

    elif degree == 7:
        slope_coeff_7 = [
            [1.00000000, -0.00000000, 0.00000000, -0.00000000],
            [1.31850776, 1.88579394, -4.51838279, 2.29888190],
            [2.23530932, -0.79002085, -2.57240008, 2.17163120],
            [2.93948292, -3.90836546, 1.60923562, 0.41682763],
            [3.56062038, -7.45667362, 7.38064952, -2.42466148],
            [4.15568002, -11.88947420, 16.25427808, -7.56136494],
            [4.46743293, -13.98559715, 20.16395287, -9.72835778],
            [4.74662438, -15.87900286, 23.66636188, -11.64087875],
            [5.00772291, -17.69353071, 27.06459723, -13.51030535],
            [5.25172323, -19.42100742, 30.32965344, -15.31627848],
            [5.48006949, -21.06112317, 33.45141064, -17.05013308],
            [5.69490550, -22.62194425, 36.43860655, -18.71459349],
            [5.89647326, -24.09995981, 39.27978104, -20.30174825],
            [6.08549612, -25.49650047, 41.97392281, -21.80988331],
            [6.26567751, -26.83613964, 44.56594166, -23.26333522]
        ]
        index_alpha = int(2*alpha + 0.5) - 2
        return slope_coeff_7[index_alpha]

    elif degree == 9:
        slope_coeff_9 = [
            [1.00000000, -0.00000000, -0.00000000, 0.00000000, -0.00000000],
            [1.33743875, 1.55213754, -3.02825657, -0.12350511, 1.28325061],
            [1.85623249, 3.82397125, -19.70879455, 26.15510902, -11.15375327],
            [2.79126397, -1.30687551, -10.57298680, 20.02623286, -9.98284231],
            [3.51036396, -6.31644952, 0.92439798, 9.32834829, -6.50264005],
            [4.15462973, -11.85851451, 16.03418150, -7.07985902, -0.31040920],
            [4.76270090, -18.23743983, 36.10529118, -31.35677926, 9.66532431],
            [5.34087782, -25.67018163, 63.87617747, -70.15437134, 27.66951403],
            [5.64305564, -28.94026159, 74.52401661, -83.54012582, 33.39343065],
            [5.92841230, -32.11619291, 85.01764165, -96.84966316, 39.11863693],
            [6.19837979, -35.18789052, 95.28157108, -109.95601312, 44.78177264],
            [6.45529995, -38.16327397, 105.31193936, -122.83169063, 50.36462504],
            [6.69888108, -41.02503190, 115.02784036, -135.35603880, 55.81014424],
            [6.92966632, -43.76867314, 124.39645141, -147.47363378, 61.09053024],
            [7.15179080, -46.43557440, 133.54648929, -159.34156394, 66.27157886]
        ]
        index_alpha = int(2*alpha + 0.5) - 2
        return slope_coeff_9[index_alpha]

    elif degree == 11:
        slope_coeff_11 = [
            [1.00000000, 0.00000000, -0.00000000, 0.00000000, -0.00000000, 0.00000000],
            [1.66678889, -3.86308014, 24.49259996, -60.31838443, 60.41749474, -21.39625618],
            [1.72148233, 5.57093260, -27.63510699, 42.17244497, -25.90619413, 5.05129251],
            [2.44812002, 5.44250700, -51.76972915, 122.80044756, -120.64444655, 42.75429141],
            [3.34207384, -1.88198559, -31.55985305, 99.72042171, -110.92549331, 42.35008523],
            [4.07315446, -9.21949162, -6.32221899, 61.04544659, -83.92082003, 35.39862577],
            [4.73684644, -17.20876074, 25.84450290, 3.70364719, -37.08075727, 21.06386836],
            [5.35950086, -26.04602434, 66.00030136, -75.00728398, 32.49211360, -1.73763297],
            [5.96254045, -36.16972228, 118.20339443, -187.72895027, 140.02594416, -39.23184984],
            [6.51682984, -46.98047500, 181.98817630, -342.88822007, 304.46069970, -102.16048673],
            [6.81927870, -51.86023914, 206.63033666, -395.90039101, 355.40642199, -120.17357682],
            [7.10938299, -56.65550129, 231.15377258, -449.06355949, 406.75991367, -138.39740356],
            [7.38624375, -61.32373225, 255.27181537, -501.66899832, 457.78188701, -156.55607004],
            [7.65057013, -65.85455478, 278.87482808, -553.40651672, 508.12674394, -174.51544666],
            [7.90659139, -70.30420539, 302.21512501, -604.77762046, 558.24999183, -192.42993273]
        ]
        index_alpha = int(2*alpha + 0.5) - 2
        return slope_coeff_11[index_alpha]
    
    else:
        raise ValueError(f"Unsupported degree: {degree}. Supported degrees are 3, 5, 7, 9, 11.")


def compute_poly_coeffs(alpha, degree):
    slope_coeffs = get_slope_coeffs(alpha, degree)
    poly_coeffs = np.zeros((degree + 1, degree + 1)) 

    for n in range(degree + 1):
        m_start = n + 1 if n % 2 == 0 else n
        for m in range(m_start, degree + 1, 2):  
            binomial = comb(m, n) 
            slope_index = (m - 1) // 2
            alternating_sign = -1 if (m - n + 1) % 2 else 1  

            poly_coeffs[n, m] = (
                alternating_sign * slope_coeffs[slope_index] * binomial
            )

    return poly_coeffs

def ace_enhance_image_poly(f, alpha, degree, omega_string):

    u = np.zeros_like(f)
    
    for channel in range(3):
        u[:,:,channel] = ace_enhance_image_poly_channel(f[:,:,channel], alpha, degree, omega_string)

    return u

def ace_enhance_image_poly_channel(f, alpha, degree, omega_string):
    f = f.astype(np.float64) / 255.0

    height, width = f.shape
    u = np.zeros_like(f)
    poly_coeffs = compute_poly_coeffs(alpha, degree)

    temp_sqr = f ** 2
    a = poly_coeffs[0, degree]

    for m in range(degree - 2, -1, -2):
        a = a * temp_sqr + poly_coeffs[0, m]

    dest = a * f
    u = dest.copy()

    for n in range(1, degree + 1):
        blurred = np.power(f, n)
        omega = compute_omega(width, height, omega_string)
        res = cv2.filter2D(blurred, -1, omega, borderType=cv2.BORDER_REFLECT)

        temp_sqr = f ** 2
        a = poly_coeffs[n, degree]
        m = degree

        while m - n >= 2:
            m -= 2
            a = a * temp_sqr + poly_coeffs[n, m]

        if n % 2 == 0:
            a *= f

        res *= a
        u += res

    u = stretch(u)
    return u

def compute_omega(width, height, omega_string):

    omega = np.zeros((height, width))
    if omega_string is None or omega_string == "1/r":
        # omega = 1/sqrt(x^2 + y^2)
        for y in range(height):
            for x in range(width):
                omega[y, x] = 0 if (x == 0 and y == 0) else 1.0 / np.sqrt(x * x + y * y)
    elif omega_string == "1":
        # omega = 1
        for y in range(height):
            for x in range(width):
                omega[y, x] = 1.0
    elif len(omega_string) >= 3 and omega_string[0] == 'G' and omega_string[1] == ':':
        # omega = Gaussian
        sigma = float(omega_string[2:])
        
        if sigma <= 0:
            return None 
        
        for y in range(height):
            for x in range(width):
                omega[y, x] = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    else:
        return None  
    
    omega /= np.sum(omega)
    
    return omega

def stretch(image):
    dest = image

    min_val = np.min(dest)
    max_val = np.max(dest)

    if min_val < max_val:
        scale = max_val - min_val
        dest[:] = ((dest - min_val) / scale) * 255

    return image.astype(np.uint8)

# def ace_enhance_image_poly_channel(f, alpha, degree, omega_string):

#     f = f.astype(np.float64) / 255.0

#     height, width = f.shape
#     num_pixels = width * height

#     u = np.zeros_like(f)
#     dest = np.zeros(num_pixels)
#     poly_coeffs = compute_poly_coeffs(alpha, degree)
    
#     src = f.flatten().copy()
#     temp_sqr = src ** 2
#     a = poly_coeffs[0, degree]

#     for m in range(degree - 2, -1, -2):
#         a = a * temp_sqr + poly_coeffs[0, m]

#     dest = a * src

#     u = dest.reshape((height, width))

#     for n in range(1, degree+1):

#         src = f.flatten().copy()
#         blurred = np.power(src, n)

#         res = convolve(blurred, compute_omega(width, height, omega_string))

#         for i in range(num_pixels):
#             Temp = src[i]
#             TempSqr = Temp ** 2
#             a = poly_coeffs[n, degree]
#             m = degree

#             while m - n >= 2:
#                 m -= 2
#                 a = a * TempSqr + poly_coeffs[n, m] 

#             if n % 2 == 0:
#                 a *= Temp

#             res[i] *= a

#         dest += res
#         u = dest.reshape((height, width)) 

#     u = stretch(u)

#     return u


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
