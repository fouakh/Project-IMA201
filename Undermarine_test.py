from Undermarine_lib import *

# f_example = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#                      [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
#                      [[128, 128, 128], [0, 128, 0], [128, 0, 128]],
#                      [[255, 0, 128], [0, 255, 255], [255, 255, 255]],
#                      [[255, 255, 255], [0, 0, 0], [128, 128, 128]]], dtype=np.uint8)

f_example = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                     [[255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 0, 128]],
                     [[128, 128, 128], [0, 128, 0], [128, 0, 128], [0, 128, 255]],
                     [[255, 0, 128], [0, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[255, 255, 255], [0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)

alpha = 2.0
degree = 9
omega_string = "1/r"

def show_image_with_values(image, ax):
    height, width, channels = image.shape
    for i in range(height):
        for j in range(width):
            ax.text(j, i, f'{image[i, j, 0]:.0f},{image[i, j, 1]:.0f},{image[i, j, 2]:.0f}', 
                    color="red", ha="center", va="center", fontsize=8)

def main():

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    axes[0].imshow(f_example)
    axes[0].set_title('Image d\'entrée')
    show_image_with_values(f_example, axes[0]) 

    enhanced_image = ace_enhance_image_poly(f_example, alpha, degree, omega_string)

    axes[1].imshow(enhanced_image)  
    axes[1].set_title('Image améliorée après ACE')
    show_image_with_values(enhanced_image, axes[1])  
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()