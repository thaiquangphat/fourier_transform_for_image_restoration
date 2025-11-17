import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from ultis import gaussianKernel

# Coordinates regions
regions = {
    "crosshair": [554, 597, 493, 535],
    "crack": [150, 315, 251, 321]
}

# Select the region want to crop out
crop_x_start, crop_x_end, crop_y_start, crop_y_end = regions["crosshair"]

# Define the crop crosshair
def cropCrossHair(blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return cropped_blurred_image

# Define the ideal synthetic crosshair
def idealCrossHair(blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    h, w = cropped_blurred_image.shape

    # Create the synthetic ideal crosshair for comparison
    ideal_crosshair = np.zeros_like(cropped_blurred_image, dtype=np.uint8)
    ideal_width, ideal_length = 3, 33  # Crosshair dimensions 33 24
    center_y, center_x = cropped_blurred_image.shape[0] // 2, cropped_blurred_image.shape[1] // 2

    intensity = 240

    # Create the horizontal line of the crosshair
    ideal_crosshair[center_y - ideal_width // 2: center_y + ideal_width // 2 + 1,
                    center_x - ideal_length // 2: center_x + ideal_length // 2 + 1] = intensity

    # Create the vertical line of the crosshair
    ideal_crosshair[center_y - ideal_length // 2: center_y + ideal_length // 2 + 1,
                    center_x - ideal_width // 2: center_x + ideal_width // 2 + 1] = intensity
    
    return ideal_crosshair

def middle_rectangle(image):
    # Check if the input image is valid
    if image is None or len(image.shape) != 2:
        raise ValueError("Input must be a valid grayscale image.")

    # Get the dimensions of the image
    height, width = image.shape

    # Create a copy of the image to avoid modifying the original
    image_with_rectangle_row = image.copy()
    image_with_rectangle_col = image.copy()

    # Calculate the middle row and column
    middle_row = height // 2
    middle_col = width // 2

    # Draw horizontal lines for the middle row
    line_thickness = 1
    cv2.line(image_with_rectangle_row, (0, middle_row - line_thickness), (width, middle_row - line_thickness), 1, 1)
    cv2.line(image_with_rectangle_row, (0, middle_row + line_thickness), (width, middle_row + line_thickness), 1, 1)

    # Draw vertical lines for the middle column
    cv2.line(image_with_rectangle_col, (middle_col - line_thickness, 0), (middle_col - line_thickness, height), 1, 1)
    cv2.line(image_with_rectangle_col, (middle_col + line_thickness, 0), (middle_col + line_thickness, height), 1, 1)

    return image_with_rectangle_row, image_with_rectangle_col

def plot_middle(image):
    # Extract the middle row
    middle_row = image[image.shape[0] // 2, :]
    middle_col = image[:, image.shape[1] // 2]

    middle_rec_row, middle_rec_col = middle_rectangle(image)

    # First bar plot
    plt.figure()
    plt.suptitle("The image line and its Bar plot")

    plt.subplot(2, 2, 1)
    plt.title("Extracted Middle Row")
    plt.imshow(middle_rec_row, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.bar(range(len(middle_row)), middle_row, color='blue')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Plot for Middle Row')

    # Second bar plot
    plt.subplot(2, 2, 3)
    plt.title("Extracted Middle Column")
    plt.imshow(middle_rec_col, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.bar(range(len(middle_col)), middle_col, color='blue')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Plot for Middle Column')

    plt.tight_layout()
    plt.savefig("image/bar_plot.png")
    plt.show()

    # Modify the list, only take the Gaussian like shape
    middle_row = [value if 12 <= index <= 30 else 0 for index, value in enumerate(middle_row)]
    middle_col = [value if 14 <= index <= 28 else 0 for index, value in enumerate(middle_col)]

    return middle_row, middle_col

def remove_center_noise(F_shift, radius_keep=10):
    h, w = F_shift.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    mask = dist <= radius_keep
    return F_shift * mask

def restoreWithGaussian(blurred_image):
    u, v = blurred_image.shape

    # Crop out the cross hair section
    cropped_blurred_image = cropCrossHair(blurred_image)
    h, w = cropped_blurred_image.shape

    # Create the ideal synthetic crosshair
    ideal_crosshair = idealCrossHair(blurred_image)

    # Apply Fourier Transform to both images
    F_blurred = fft2(cropped_blurred_image)
    F_ideal = fft2(ideal_crosshair)

    # Shift the zero frequency component to the center
    F_blurred_shifted = fftshift(F_blurred)
    F_ideal_shifted = fftshift(F_ideal)

    # After fftshift of cropped crosshair
    F_blurred_shifted = remove_center_noise(F_blurred_shifted, radius_keep=7)

    # Compute the blurring function H(u,v)
    K = 0.0001
    F_shift_abs2 = np.abs(F_ideal_shifted)**2
    H_uv = (F_blurred_shifted.astype(int) * F_ideal_shifted.conj().astype(int)) / (F_shift_abs2.astype(int) + K)

    middle_row, middle_col = plot_middle(np.abs(H_uv))

    # Calculate estimated D0
    h_center, w_center = h//2, w//2
    D0, counter = 0.0, 0

    for j in range(w):
        if middle_row[j] > 0:
            dist = (j - w_center)**2
            huv = np.log(middle_row[j])
            d0 = np.sqrt(-dist / (2 * huv))
            D0 += d0

            counter += 1

    for i in range(h):
        if middle_col[i] > 0:
            dist = (i - h_center)**2
            huv = np.log(middle_col[i])
            d0 = np.sqrt(-dist / (2 * huv))
            D0 += d0    

            counter += 1

    D0 = D0 / (counter)
    print(f"Estimated D0: {D0}")

    # Define the estimate gaussian kernel
    G_shift = gaussianKernel(h, w, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.0001 # Regularization parameter
    H_uv_abs2 = np.abs(G_shift)**2
    restored_fft = (F_blurred_shifted * G_shift.conj()) / (H_uv_abs2 + K)

    # Inverse FFT to restore the image cross
    restored_image = np.abs(ifft2(ifftshift(restored_fft)))

    plt.figure(figsize=(9,3))
    plt.suptitle("Ideal Crosshair and Its Restoration")

    plt.subplot(1, 3, 1)
    plt.title("Ideal Crosshair")
    plt.imshow(ideal_crosshair, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Blurred Crosshair")
    plt.imshow(cropped_blurred_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Restored Crosshair")
    plt.imshow(restored_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('image/restore_crosshair.jpg')

    # Update D0 for entire image
    D0 = D0 * (u / w)
    print(f"Scaled coefficient k: {u/h}")

    # Restore the image
    F_original_fft = fft2(blurred_image)
    F_original_fftshift = fftshift(F_original_fft)

    G_full_shift = gaussianKernel(u, v, D0)

    # Regularized deblurring (Wiener filter approach)
    K = 0.005 # Regularization parameter
    H_uv_abs2 = np.abs(G_full_shift)**2
    restore = (F_original_fftshift * G_full_shift.conj()) / (H_uv_abs2 + K)

    f_restore = np.abs(ifft2(ifftshift(restore)))

    f_restore = cv2.normalize(f_restore, None, 0, 255, cv2.NORM_MINMAX)
    f_restore = np.uint8(f_restore)  # Convert to uint8 for displaying

    return f_restore