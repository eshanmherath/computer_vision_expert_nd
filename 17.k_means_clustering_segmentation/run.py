import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/monarch.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# Reshape image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1, 3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# you can change the number of max iterations for faster convergence!
ITERATIONS = 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, ITERATIONS, 0.2)

# Select a k
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)

segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)
plt.show()

# Visualize one segment, try to find which is the leaves, background, etc!
plt.imshow(labels_reshape == 0, cmap='gray')
plt.show()
plt.imshow(labels_reshape == 1, cmap='gray')
plt.show()

# mask an image segment by cluster
cluster = 2
masked_image = np.copy(image)

# turn the mask green!
masked_image[labels_reshape == cluster] = [0, 255, 0]
plt.imshow(masked_image)
plt.show()

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))

masked_image0 = np.copy(image)
masked_image0[labels_reshape == 0] = [0, 255, 0]
masked_image1 = np.copy(image)
masked_image1[labels_reshape == 1] = [0, 255, 0]
masked_image2 = np.copy(image)
masked_image2[labels_reshape == 2] = [0, 255, 0]

ax1.set_title('Original Image')
ax1.imshow(image)
ax2.set_title('Cluster 0')
ax2.imshow(masked_image0)
ax3.set_title('Cluster 1')
ax3.imshow(masked_image1)
ax4.set_title('Cluster 2')
ax4.imshow(masked_image2)

plt.show()
