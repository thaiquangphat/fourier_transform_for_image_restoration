import numpy as np
import cv2

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE contrast enhancement.
    image: np.uint8 grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced
