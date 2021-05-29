import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/brain_MR.jpg')
image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
plt.show()

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('Original gray')
ax1.imshow(gray, cmap='gray')

ax2.set_title('Gaussian Blurred image')
ax2.imshow(gray_blur, cmap='gray')

plt.show()

# High-pass filter

# 3x3 sobel filters for edge detection
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Filter the original and blurred grayscale images using filter2D
filtered = cv2.filter2D(gray, -1, sobel_x)

filtered_blurred = cv2.filter2D(gray_blur, -1, sobel_x)
f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('Sobel-X On Gray')
ax1.imshow(filtered, cmap='gray')

ax2.set_title('Sobel-X Gaussian Blurred Gray')
ax2.imshow(filtered_blurred, cmap='gray')
plt.show()

retval, binary_image = cv2.threshold(filtered_blurred, 50, 255, cv2.THRESH_BINARY)

plt.imshow(binary_image, cmap='gray')
plt.show()
