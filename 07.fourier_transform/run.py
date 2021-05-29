import cv2
import matplotlib.pyplot as plt
import numpy as np

image_stripes = cv2.imread('images/stripes.jpg')
image_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_BGR2RGB)

image_solid = cv2.imread('images/pink_solid.jpg')
image_solid = cv2.cvtColor(image_solid, cv2.COLOR_BGR2RGB)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(image_stripes)
ax2.imshow(image_solid)

plt.show()

gray_strips = cv2.cvtColor(image_stripes, cv2.COLOR_RGB2GRAY)
gray_solid = cv2.cvtColor(image_solid, cv2.COLOR_RGB2GRAY)

# Normalize image color values from range [0, 255] to [0,1]

norm_stripes = gray_strips / 255.0
norm_solid = gray_solid / 255.0


def ft_image(norm_image):
    """
    :param norm_image: takes in a normalized, grayscale image
    :return: a frequency spectrum transform of that image
    """
    f = np.fft.fft2(norm_image)
    f_shift = np.fft.fftshift(f)
    frequency_tx = 20 * np.log(np.abs(f_shift))

    return frequency_tx


f_stripes = ft_image(norm_stripes)
f_solid = ft_image(norm_solid)

f2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

ax1.set_title('Original Image')
ax1.imshow(image_stripes)
ax2.set_title('Frequency transform image')
ax2.imshow(f_stripes, cmap='gray')

ax3.set_title('Original image')
ax3.imshow(image_solid)
ax4.set_title('Frequency transform image')
ax4.imshow(f_solid, cmap='gray')

plt.show()

image = cv2.imread('images/birds.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

norm_image = gray / 255.0

f_image = ft_image(norm_image)

f3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(image)
ax2.imshow(f_image, cmap='gray')

plt.show()

# Notice that this image has components of all frequencies. You can see a bright spot in the center of the
# transform image, which tells us that a large portion of the image is low-frequency; this makes sense since
# the body of the birds and background are solid colors. The transform image also tells us that there are two
# dominating directions for these frequencies; vertical edges (from the edges of birds) are represented by a horizontal
# line passing through the center of the frequency transform image, and horizontal edges (from the branch and
# tops of the birds' heads) are represented by a vertical line passing through the center
