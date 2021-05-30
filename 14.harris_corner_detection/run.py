import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/waffle.jpg')

image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
plt.show()

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)  # Because Harris works with floats

PIXEL_WIDTH = 2
KERNEL_SIZE = 3
k = 0.04

dst = cv2.cornerHarris(gray, PIXEL_WIDTH, KERNEL_SIZE, k)

plt.imshow(dst, cmap='gray')
plt.show()

# Dilate corner image to enhance corner points
dst = cv2.dilate(dst, None)

plt.imshow(dst, cmap='gray')
plt.show()

# Extract and display strong corners

PERCENTAGE = 0.1  # Lowering this would color more corners (NOT detect; detection is already done at this point)
thresh = PERCENTAGE * dst.max()

corner_image = np.copy(image_copy)

GREEN = (0, 255, 0)
LINE_THICKNESS = 1

for j in range(0, dst.shape[0]):
    for i in range(0, dst.shape[1]):
        if dst[j, i] > thresh:
            cv2.circle(corner_image, (i, j), 1, GREEN, LINE_THICKNESS)

plt.imshow(corner_image, cmap='gray')
plt.show()
