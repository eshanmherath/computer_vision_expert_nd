import cv2
import matplotlib.pyplot as plt
import numpy as np

import helpers

standard_w = 1100
standard_h = 600


def _standardize_input(image):
    # Resize image so that all "standard" images are the same size 600x1100 (hxw)
    resized_image = cv2.resize(image, (1100, 600))  # CV2 takes (w, h)

    return resized_image


def _encode(label):
    encoded_value = 0
    if label == "day":
        encoded_value = 1

    return encoded_value


def standardize(image_list):
    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]

        standardized_im = _standardize_input(image)

        binary_label = _encode(label)

        standard_list.append((standardized_im, binary_label))

    return standard_list


# Expects a RGB image
def _average_brightness(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    brightness = np.sum(hsv[:, :, 2])

    total_pixels = standard_w * standard_h

    avg = brightness / total_pixels

    return avg


image_dir_train = 'day_night_images/training'
image_dir_test = 'day_night_images/test'

IMAGE_LIST = helpers.load_dataset(image_dir_train)

# Visualize one image

image_index = 0
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]

print("Image Shape", selected_image.shape)
print("Image Label", selected_label)

plt.imshow(selected_image)
plt.show()

selected_image_hsv = cv2.cvtColor(selected_image, cv2.COLOR_RGB2HSV)

h = selected_image_hsv[:, :, 0]
s = selected_image_hsv[:, :, 1]
v = selected_image_hsv[:, :, 2]

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
ax1.set_title('Standardized image')
ax1.imshow(selected_image)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

plt.show()

STANDARDIZED_LIST = standardize(IMAGE_LIST)

image_num = 190
test_im = STANDARDIZED_LIST[image_num][0]

avg = _average_brightness(test_im)
print('Avg brightness: ' + str(avg))
plt.imshow(test_im)

# Display a night image using image data

day_images_pixels = [_average_brightness(image[0]) for image in STANDARDIZED_LIST if image[1] == 1]

average_day_images = np.average(day_images_pixels)
minimum_day_image = np.min(day_images_pixels)
maximum_day_image = np.max(day_images_pixels)

night_images_pixels = [_average_brightness(image[0]) for image in STANDARDIZED_LIST if image[1] == 0]
average_night_images = np.average(night_images_pixels)
minimum_night_images = np.min(night_images_pixels)
maximum_night_images = np.max(night_images_pixels)

print("\naverage_day_images", average_day_images)
print("minimum_day_image", minimum_day_image)
print("maximum_day_image", maximum_day_image)
print("\naverage_night_images", average_night_images)
print("minimum_night_images", minimum_night_images)
print("maximum_night_images", maximum_day_image)

# Crude method. There will be night images not captured
for image in STANDARDIZED_LIST:
    if _average_brightness(image[0]) <= minimum_day_image:
        night_image = image[0]
        plt.imshow(night_image)
        plt.show()
