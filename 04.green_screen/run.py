import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('images/car_green_screen.jpg')
# image = mpimg.imread('images/car_green_screen2.jpg')

print('Image Dimensions', image.shape)

height = image.shape[0]
width = image.shape[1]

plt.imshow(image)
plt.show()
image_copy = np.copy(image)

lower_green = np.array([0, 200, 0])
upper_green = np.array([250, 255, 250])

mask = cv2.inRange(image_copy, lower_green, upper_green)

plt.imshow(mask, cmap='gray')
plt.show()

masked_image = np.copy(image)

masked_image[mask != 0] = [0, 0, 0]

plt.imshow(masked_image)
plt.show()

background_image = mpimg.imread('images/sky.jpg')
background_image_copy = np.copy(background_image)  # This or some other way to make the image writable.
# Otherwise it will fail when applying mask below.
# This is not an issue if just after reading, a BGR to RGB conversion is there.

crop_background = background_image_copy[0:height, 0:width]

plt.imshow(crop_background)
plt.show()

crop_background[mask == 0] = [0, 0, 0]

plt.imshow(crop_background)
plt.show()

complete_image = masked_image + crop_background

plt.imshow(complete_image)
plt.show()
