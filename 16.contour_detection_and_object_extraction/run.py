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


def orientations(contours):
    """
        Orientation
        :param contours: a list of contours
        :return: angles, the orientations of the contours
    """

    angles = []

    for cnt in contours:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        angles.append(angle)

    return angles


print(f"Number of Contours: {len(contours)}")  # This should ideally be 2 since we have two hands on the image
angles = orientations(contours)
print('Angles of each contour (in degrees): ' + str(angles))


def left_hand_crop(image, selected_contour):
    """
        Left hand crop
        :param image: the original image
        :param selected_contour: the contour that will be used for cropping
        :return: cropped_image, the cropped image around the left hand
    """

    cropped_image = np.copy(image)

    x, y, w, h = cv2.boundingRect(selected_contour)

    cropped_image = cropped_image[y: y + h, x: x + w]

    return cropped_image


# From the two contour values
# [61.35833740234375, 82.27550506591797]
# One closest to 90 can be taken as the left hand since it has a more upward position
selected_contour = contours[1]

if selected_contour is not None:
    cropped_image = left_hand_crop(image, selected_contour)
    plt.imshow(cropped_image)
    plt.show()
