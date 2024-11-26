import cv2
import matplotlib.pyplot as plt

def main():
    image_paths_1 = [
        "Images/image001.jpg",
        "Images/image002.jpg",
        "Images/image010.png",
        "Images/image016.png"
    ]

    image_paths_2 = [
        "Images_RGB_WP_20p/EX_image001.jpg",
        "Images_RGB_WP_20p/EX_image002.jpg",
        "Images_RGB_WP_20p/EX_image010.png",
        "Images_RGB_WP_20p/EX_image016.png"
    ]

    images_1 = [cv2.imread(path) for path in image_paths_1]
    images_2 = [cv2.imread(path) for path in image_paths_2]

    fig, axes = plt.subplots(2, 4, figsize=(25, 10))

    for i in range(4):
        axes[0, i].imshow(cv2.cvtColor(images_1[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Image Originale {i+1}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(cv2.cvtColor(images_2[i], cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Image RGB White Patches {i+1}", fontsize=8)
        axes[1, i].axis("off")

    plt.subplots_adjust(hspace=0.2) 

    plt.show()

if __name__ == '__main__':
    main()
