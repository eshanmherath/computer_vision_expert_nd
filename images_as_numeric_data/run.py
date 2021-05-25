import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread("images/waymo_car.jpg")

print("Image Dimensions", image.shape)  # Height, Width, Color Channels

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')
plt.show()

x = 400
y = 300

print(gray_image[y, x])

max_val = np.amax(gray_image)
min_val = np.amin(gray_image)

print('Max: ', max_val)
print('Min: ', min_val)

tiny_image = np.array([
    [0, 20, 30, 150, 120],
    [200, 200, 250, 70, 3],
    [50, 180, 85, 40, 90],
    [240, 100, 50, 255, 10],
    [30, 0, 75, 190, 220]
])

plt.imshow(tiny_image, cmap='gray')
plt.show()

tiny_face_image = np.array([
    [0, 0, 0, 0, 0],
    [0, 200, 0, 200, 0],
    [0, 0, 0, 0, 0],
    [200, 0, 0, 0, 200],
    [0, 200, 200, 200, 0]
])

plt.imshow(tiny_face_image, cmap='gray')
plt.show()
