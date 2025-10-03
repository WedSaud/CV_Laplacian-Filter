import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image as grayscale
img = cv2.imread('moon.tif', 0)

# Display original image
plt.figure(figsize=(8,5), dpi=150)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# Define the Laplacian kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

# Apply Laplacian filter
LaplacianImage = cv2.filter2D(src=img,
                              ddepth=cv2.CV_64F,  # Use 64-bit floating point for better precision
                              kernel=kernel)

plt.figure(figsize=(8,5), dpi=150)
plt.imshow(LaplacianImage, cmap='gray')
plt.axis('off')
plt.show()

# Set the constant value for sharpening
c = -1

# Perform sharpening operation using higher precision (int16 or float)
g = img.astype(np.float64) + c * LaplacianImage

# Clip the result to stay within valid pixel range
gClip = np.clip(g, 0, 255)

# Convert the result back to uint8 for proper display
gClip = gClip.astype(np.uint8)

# Display the sharpened image
plt.figure(figsize=(8,5), dpi=150)
plt.imshow(gClip, cmap='gray')
plt.axis('off')
plt.show()
