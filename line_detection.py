import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, filters, morphology, measure

def extract_boundary_morphology(img_gray):
    """Approach 1: Morphology + Largest Component Boundary."""
    # Blur to suppress small vessels
    blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # Sobel gradient
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold using Otsu
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to join edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component
    num_labels, labels = cv2.connectedComponents(closed)
    if num_labels <= 1:
        return np.zeros_like(img_gray)

    largest_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
    mask = (labels == largest_label).astype(np.uint8)

    # Extract contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if contours:
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

    return canvas


def extract_boundary_chanvese(img_gray):
    """Approach 2: Chan–Vese Active Contours."""
    # Normalize to [0, 1]
    img_norm = img_gray.astype(np.float32) / 255.0

    # Run Chan–Vese
    cv = segmentation.chan_vese(
        img_norm,
        mu=0.25,
        lambda1=1,
        lambda2=1,
        max_num_iter=200
    )


    # Extract boundary mask
    boundary = segmentation.find_boundaries(cv, mode="thick")

    # Overlay
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay[boundary] = [0, 0, 255]  # red
    return overlay


def line_detections(image_list, gradient_kernel=3, hough_threshold=100):
    n_images = len(image_list)
    fig, axes = plt.subplots(n_images, 5, figsize=(18, 4 * n_images))

    if n_images == 1:
        axes = np.expand_dims(axes, 0)

    for idx, img in enumerate(image_list):
        # Ensure grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        # --- Compute gradient magnitude ---
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=gradient_kernel)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=gradient_kernel)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255,
                                           cv2.NORM_MINMAX).astype(np.uint8)

        # --- Canny edges for Hough ---
        edges = cv2.Canny(gradient_magnitude, 150, 200)

        # --- Hough line detection ---
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=hough_threshold,
                                minLineLength=30, maxLineGap=5)

        img_lines = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # --- New Method 1: Morphology Boundary ---
        morph_boundary = extract_boundary_morphology(img_gray)

        # --- New Method 2: Chan–Vese Boundary ---
        chan_boundary = extract_boundary_chanvese(img_gray)

        # --- Plot everything ---
        axes[idx, 0].imshow(img_gray, cmap='gray')
        axes[idx, 0].set_title('Original')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(gradient_magnitude, cmap='gray')
        axes[idx, 1].set_title('Gradient Magnitude')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title('Hough Lines')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(cv2.cvtColor(morph_boundary, cv2.COLOR_BGR2RGB))
        axes[idx, 3].set_title('Morphology Boundary')
        axes[idx, 3].axis('off')

        axes[idx, 4].imshow(cv2.cvtColor(chan_boundary, cv2.COLOR_BGR2RGB))
        axes[idx, 4].set_title('Chan–Vese Boundary')
        axes[idx, 4].axis('off')

    plt.tight_layout()
    plt.show()
