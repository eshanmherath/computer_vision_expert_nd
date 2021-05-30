import cv2
import matplotlib.pyplot as plt

# Sunflower

image = cv2.imread('images/sunflower.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Better to have 1:2 or 1:3 ratio for lower:upper
lower = 100’
upper = 200

edges = cv2.Canny(gray, lower, upper)

plt.figure(figsize=(20, 10))
plt.imshow(edges, cmap='gray')
plt.show()

# Brain MR

image = cv2.imread('images/brain_MR.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')
plt.show()

# Canny Thresholds
wide = cv2.Canny(gray, 30, 100)
tight = cv2.Canny(gray, 200, 240)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')

ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')
plt.show()
