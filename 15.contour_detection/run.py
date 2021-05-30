import cv2
import matplotlib.pyplot as plt

import numpy as np

image = cv2.imread('images/thumbs_up_down.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)  # INV get the inverse (white in black background)

plt.imshow(binary, cmap='gray')
plt.show()

# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

GREEN = (0, 255, 0)
LINE_THICKNESS = 3

contours_image = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, GREEN, LINE_THICKNESS)

plt.imshow(contours_image, cmap='gray')
plt.show()
