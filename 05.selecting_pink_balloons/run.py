import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/water_balloons.jpg')

plt.imshow(image)
plt.show()

# Convert from BRG to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# RGB Channels

r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

ax1.set_title('Red')
ax1.imshow(r, cmap='gray')

ax2.set_title('Green')
ax2.imshow(g, cmap='gray')

ax3.set_title('Blue')
ax3.imshow(b, cmap='gray')

plt.show()

# Convert from RGB to HSV

hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

h = hsv_image[:, :, 0]
s = hsv_image[:, :, 1]
v = hsv_image[:, :, 2]

f2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(20, 10))

ax4.set_title('Hue')
ax4.imshow(h, cmap='gray')

ax5.set_title('Saturation')
ax5.imshow(s, cmap='gray')

ax6.set_title('Value')
ax6.imshow(v, cmap='gray')

plt.show()

# Pink and Hue Thresholds

lower_pink = np.array([180, 0, 100])
upper_pink = np.array([255, 255, 255])

lower_hue = np.array([160, 0, 0])
upper_hue = np.array([180, 255, 255])  # Note: Hue range is 0-180 (not 0-255)

# Creating Pink Mask

mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

plt.imshow(mask_rgb, cmap='gray')
plt.show()

masked_image = np.copy(image)
masked_image[mask_rgb == 0] = [0, 0, 0]

plt.imshow(masked_image, cmap='gray')
plt.show()

# Creating Hue Mask

mask_hsv = cv2.inRange(hsv_image, lower_hue, upper_hue)
plt.imshow(mask_hsv, cmap='gray')
plt.show()

masked_image_hsv = np.copy(image)
masked_image_hsv[mask_hsv == 0] = [0, 0, 0]

plt.imshow(masked_image_hsv, cmap='gray')
plt.show()

f3, (ax7, ax8, ax9, ax10) = plt.subplots(1, 4, figsize=(20, 10))

ax7.set_title('Pink Mask')
ax7.imshow(mask_rgb, cmap='gray')

ax8.set_title('Pink Balloons (Pink Mask)')
ax8.imshow(masked_image, cmap='gray')

ax9.set_title('Hue Mask')
ax9.imshow(mask_hsv, cmap='gray')

ax10.set_title('Pink Balloons (Hue Mask)')
ax10.imshow(masked_image_hsv, cmap='gray')

plt.show()
