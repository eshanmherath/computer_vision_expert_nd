import matplotlib.pyplot as plt
import numpy as np

import helpers

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

# Display a night image using label
night_image_index = None
for index, image in enumerate(IMAGE_LIST):
    if image[1] == 'night':
        night_image = image[0]
        plt.imshow(night_image)
        plt.show()
        night_image_index = index
        break


# Display a night image using image data

def _average_pixel(im):
    return np.average(im)


day_images_pixels = [_average_pixel(image[0]) for image in IMAGE_LIST if image[1] == 'day']
average_day_images = np.average(day_images_pixels)
minimum_day_image = np.min(day_images_pixels)
maximum_day_image = np.max(day_images_pixels)

night_images_pixels = [_average_pixel(image[0]) for image in IMAGE_LIST if image[1] == 'night']
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
for image in IMAGE_LIST:
    if _average_pixel(image[0]) <= minimum_day_image:
        night_image = image[0]
        plt.imshow(night_image)
        plt.show()
