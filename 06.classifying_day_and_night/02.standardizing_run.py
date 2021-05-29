import cv2
import matplotlib.pyplot as plt

import helpers

image_dir_train = 'day_night_images/training'
image_dir_test = 'day_night_images/test'

IMAGE_LIST = helpers.load_dataset(image_dir_train)

print("Training images count", len(IMAGE_LIST))


def standardize_input(image):
    # Resize image so that all "standard" images are the same size 600x1100 (hxw)
    resized_image = cv2.resize(image, (1100, 600))  # CV2 takes (w, h)

    return resized_image


def encode(label):
    encoded_value = 0
    if label == "day":
        encoded_value = 1

    return encoded_value


def standardize(image_list):
    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]

        standardized_im = standardize_input(image)

        binary_label = encode(label)

        standard_list.append((standardized_im, binary_label))

    return standard_list


STANDARDIZED_LIST = standardize(IMAGE_LIST)

image_num = 0

selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

print(f"Image size {selected_image.shape} with label {selected_label} [Day=1; Night =0]")

plt.imshow(selected_image)
plt.show()
