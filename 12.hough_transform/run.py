import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/phone.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define our parameters for Canny
low_threshold = 50
high_threshold = 100
edges = cv2.Canny(gray, low_threshold, high_threshold)

plt.imshow(edges, cmap='gray')
plt.show()

# Hough Transform parameters

rho = 1  # Granularity of Rho
theta = np.pi / 180  # Granularity of Theta
threshold = 60
min_line_length = 50  # Length of a segment to be considered as a line
max_line_gap = 5  # Gap between two lines to be considered separate, in pixel

line_image = np.copy(image)

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

RED = (255, 0, 0)
LINE_THICKNESS = 5

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), RED, LINE_THICKNESS)

plt.imshow(line_image)
plt.show()

# With longer line
min_line_length = 110  # Length of a segment to be considered as a line

line_image = np.copy(image)

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

RED = (255, 0, 0)
LINE_THICKNESS = 5

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), RED, LINE_THICKNESS)

plt.imshow(line_image)
plt.show()
