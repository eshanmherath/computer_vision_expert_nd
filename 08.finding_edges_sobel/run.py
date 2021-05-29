import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('images/curved_lane.jpg')
# image = mpimg.imread('images/white_lines.jpg')

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

filtered_image_y = cv2.filter2D(gray, -1, sobel_y)

plt.imshow(filtered_image_y, cmap='gray')
plt.show()

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 2, 1]])

filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

plt.imshow(filtered_image_x, cmap='gray')
plt.show()
