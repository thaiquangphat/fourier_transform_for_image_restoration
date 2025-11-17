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
def idealCrossHair(cropped_blurred_image, blurred_image):
    # Crop the region to be compared with the restored version
    cropped_blurred_image = blurred_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    h, w = cropped_blurred_image.shape

    # Create the synthetic ideal crosshair for comparison
    ideal_crosshair = np.zeros_like(cropped_blurred_image, dtype=np.uint8)
    ideal_width, ideal_length = 3, 33  # Crosshair dimensions
    center_y, center_x = cropped_blurred_image.shape[0] // 2, cropped_blurred_image.shape[1] // 2

    intensity = 255

    # Create the horizontal line of the crosshair
    ideal_crosshair[center_y - ideal_width // 2: center_y + ideal_width // 2 + 1,
                    center_x - ideal_length // 2: center_x + ideal_length // 2 + 1] = intensity

    # Create the vertical line of the crosshair
    ideal_crosshair[center_y - ideal_length // 2: center_y + ideal_length // 2 + 1,
                    center_x - ideal_width // 2: center_x + ideal_width // 2 + 1] = intensity
    
    return ideal_crosshair

if __name__ == "__main__":
    blurred_image_input = cv2.imread("image/heart.jpg", cv2.IMREAD_GRAYSCALE)
    cropped_cross  = cropCrossHair(blurred_image_input)
    ideal_crosshair = idealCrossHair(cropped_cross, blurred_image_input)

    h, w = cropped_cross.shape
    lowpass = gaussianKernel(h, w, 2.22767338548629)

    G_fft = fft2(cropped_cross)
    F_fft = fft2(ideal_crosshair)

    G_shift = fftshift(G_fft)
    F_shift = fftshift(F_fft)

    plt.figure(figsize=(9, 7))
    plt.suptitle("Fourier Spectrum of Crosshair versions")

    plt.subplot(2, 3, 1)
    plt.title("Fourier of Ideal")
    plt.imshow(cv2.normalize(np.abs(F_shift), 0, 255, cv2.NORM_MINMAX), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Fourier of Blurred")
    plt.imshow(cv2.normalize(np.abs(G_shift), 0, 255, cv2.NORM_MINMAX), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Fourier of Estimated")
    plt.imshow(cv2.normalize(np.abs(F_shift * lowpass), 0, 255, cv2.NORM_MINMAX), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Ideal Cross")
    plt.imshow(ideal_crosshair, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Cropped of Blurred")
    plt.imshow(cropped_cross, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Blurred estimated with Gaussian")
    plt.imshow(np.abs(ifft2(ifftshift(F_shift * lowpass))).astype(int), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("image/fourier_crosshair.jpg")

    # Plot out G / F
    K = 0.0001
    F_shift_abs2 = np.abs(F_shift)**2
    H_shift = (G_shift.astype(int) * F_shift.conj().astype(int)) / (F_shift_abs2.astype(int) + K)

    N_shift = G_shift - F_shift * lowpass

    K = 0.005 # Regularization parameter
    H_uv_abs2 = np.abs(lowpass)**2
    restore = (G_shift * lowpass.conj()) / (H_uv_abs2 + K)

    f_restore = np.abs(ifft2(ifftshift(restore)))
    f_restore = cv2.normalize(f_restore, None, 0, 255, cv2.NORM_MINMAX)
    f_restore = np.uint8(f_restore)  # Convert to uint8 for displaying

    plt.figure(figsize=(5, 3))   

    plt.suptitle("Fourier Spectrum of Gaussian and Estimated function")

    plt.subplot(1, 2, 1)
    plt.title("Fourier of Gaussian")
    plt.imshow(lowpass, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Fourier of Degraded Function")
    plt.imshow(cv2.normalize(np.abs(H_shift), 0, 255, cv2.NORM_MINMAX), cmap='gray')
    plt.axis('off') 

    plt.tight_layout()
    plt.savefig('image/fourier_spectrum_degfunc.jpg')
    
    plt.show()
