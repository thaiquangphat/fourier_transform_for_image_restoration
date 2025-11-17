import cv2
import numpy as np
import matplotlib.pyplot as plt

def plotHistogram(image):
    # Optinal: select a segmented region to observe histogram
    x_start, x_end = 457, 482
    y_start, y_end = 506, 535

    image = image[x_start:x_end, y_start:y_end]

    # Calculate histograms
    original_histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    # Plot the original and noisy images along with their histograms
    plt.figure(figsize=(6, 3))

    # Original image and its histogram
    plt.subplot(1, 2, 1)
    plt.title("(Segmented) Image")
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Histogram")
    plt.bar(bin_edges[:-1], original_histogram, width=1, edgecolor='black')
    plt.xlim(0, 255)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load an image using OpenCV
    image_path = 'image/heart.jpg'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to read!")
    
    plotHistogram(image)
