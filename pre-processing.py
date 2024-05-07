import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# Load image
img_path = 'C:/Users/rober/PycharmProjects/Handwritten-Recognition/JustBinarisedImages/P583-Fg006-R-C01-R01-binarized.jpg'
image = cv2.imread(img_path, cv2.IMREAD_COLOR)

"""
# Display the image
if image is None:
    print("Error: Unable to load image.")
else:
    # Convert color channels from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a window that resizes according to the image size
    cv2.namedWindow('Binarized Dead Sea Scroll Example', cv2.WINDOW_NORMAL)

    # Display the image
    cv2.imshow('Binarized Dead Sea Scroll Example', image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

# Pre-process image (Inverting the image to a binary image and tresholding it with Otsu's method)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
inverted_img = cv2.bitwise_not(th)

"""
# Display the pre-processed image
if inverted_img is None:
    print("Error: Unable to load image.")
else:

    # Create a window that resizes according to the image size
    cv2.namedWindow('Pre-processed Dead Sea Scroll Example', cv2.WINDOW_NORMAL)

    # Display the image
    cv2.imshow('Pre-processed Dead Sea Scroll Example', inverted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

# Create a directory to save the morphological operation images
output_dir = 'MorphologicalOperationImages'
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f'{output_dir}/Preprocessed_Image.jpg', inverted_img)

# Define different structuring elements for morphological operations
kernels = [
    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),  # Small Rectangular kernel
    cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)),  # Medium Rectangular kernel
    cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),  # Big Rectangular kernel

    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),  # Small Elliptical kernel
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)),  # Medium Elliptical kernel
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),  # Big Elliptical kernel

    cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),  # Small Cross-shaped kernel
    cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6)),  # Medium Cross-shaped kernel
    cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)),  # Big Cross-shaped kernel
]

# Apply morphological operations with different kernels and save the results
for i, kernel in enumerate(kernels):
    # Dilation
    dilated_img = cv2.dilate(inverted_img, kernel, iterations=1)
    cv2.imwrite(f'{output_dir}/Dilation_Image_{i}.jpg', dilated_img)

    # Erosion
    eroded_img = cv2.erode(inverted_img, kernel, iterations=1)
    cv2.imwrite(f'{output_dir}/Erosion_Image_{i}.jpg', eroded_img)

    # Opening
    opened_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f'{output_dir}/Opening_Image_{i}.jpg', opened_img)

    # Closing
    closed_img = cv2.morphologyEx(inverted_img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f'{output_dir}/Closing_Image_{i}.jpg', closed_img)

# It seemed like Closing produced the cleanest outputs.
# Specifically, the closing morphological operation with a "big" ellptical structuring element.