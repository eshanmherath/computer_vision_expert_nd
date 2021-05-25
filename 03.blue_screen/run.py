import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/pizza_bluescreen.jpg')

print('Image Type', type(image))
print("Image Dimensions", image.shape)

height = image.shape[0]
width = image.shape[1]

image_copy = np.copy(image)
plt.imshow(image_copy)
plt.show()

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)
plt.show()

# Change value below to get the best mask
lower_blue = np.array([0, 0, 220])
upper_blue = np.array([250, 250, 255])

mask = cv2.inRange(image_copy, lower_blue, upper_blue)

plt.imshow(mask, cmap='gray')
plt.show()

masked_image = np.copy(image_copy)

masked_image[mask != 0] = [0, 0, 0]

plt.imshow(masked_image)
plt.show()

background_image = cv2.imread('images/space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

crop_background = background_image[0: height, 0: width]

crop_background[mask == 0] = [0, 0, 0]

plt.imshow(crop_background)
plt.show()

# Addition will work since the covered parts of each image is having 0s
complete_image = masked_image + crop_background

plt.imshow(complete_image)
plt.show()
