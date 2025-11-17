import numpy as np
import cv2
import matplotlib.pyplot as plt

# Coordinates regions
regions = {
    "crosshair": [554, 597, 493, 535],
    "crack": [150, 315, 251, 321]
}

# Select the region to crop
crop_x_start, crop_x_end, crop_y_start, crop_y_end = regions["crack"]

# Define the crop function
def cropImage(image):
    # Crop the region
    cropped_image = image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    return cropped_image

if __name__ == "__main__":
    # Read images in color (BGR)
    input_img = cv2.imread("image/heart.jpg", cv2.IMREAD_COLOR)
    restored = cv2.imread("image/restore.jpg", cv2.IMREAD_COLOR)
    bright = cv2.imread("image/brightened.jpg", cv2.IMREAD_COLOR)
    sharp = cv2.imread("image/sharpened.jpg", cv2.IMREAD_COLOR)

    # Crop regions
    input_cross = cropImage(input_img)
    restore_cross = cropImage(restored)
    bright_cross = cropImage(bright)
    sharpen_cross = cropImage(sharp)

    plt.figure(figsize=(8, 6))
    plt.suptitle("Cropped Cracks", fontsize=16, y=0.9)

    images = [input_cross, restore_cross, bright_cross, sharpen_cross]
    titles = ["Input Blurred Image", "Restored Image", "Brighten Restored Image", "Sharpened Restored Image"]

    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 2, i)
        # Convert BGR (OpenCV) to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("image/crack_four.png")
    plt.show()
