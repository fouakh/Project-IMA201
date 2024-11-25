import cv2
import matplotlib.pyplot as plt
from Undermarine_lib import ace_enhance_image_poly
import numpy as np

def plot_histogram_rgb(image, ax, title):
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        ax.bar(range(256), hist, color=color, alpha=0.3, width=2)  # Épaisseur des bins augmentée avec width=2
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xticks([])  # Supprime les numéros des axes X
    ax.set_yticks([])  # Supprime les numéros des axes Y

def main():
    image_paths = [
        "Images_Test_for_ACE/imageTest002.png",
        "Images_Test_for_ACE/imageTest003.png",
        "Images_Test_for_ACE/imageTest004.png"
    ]

    images = [cv2.imread(path) for path in image_paths]

    for i, img in enumerate(images):
        if img is None:
            raise ValueError(f"Impossible de charger l'image : {image_paths[i]}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.subplots_adjust(hspace=1.5)
    fig.suptitle("Résultats ACE", fontsize=16)

    for i, img in enumerate(images):
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Image originale {i+1}")
        axes[i, 0].axis("off")

        plot_histogram_rgb(img, axes[i, 1], f"Histogramme original {i+1}")

        enhanced_img = ace_enhance_image_poly(img, alpha=7, degree=11, omega_string='1/r')
        axes[i, 2].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f"Image ACE {i+1}")
        axes[i, 2].axis("off")

        plot_histogram_rgb(enhanced_img, axes[i, 3], f"Histogramme ACE {i+1}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()
