import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('images/multi_faces.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20, 10))
plt.imshow(gray, cmap='gray')
plt.show()

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, 4, 6)

print('We found ' + str(len(faces)) + ' faces in this image')
print("Their coordinates and lengths/widths are as follows")
print('=============================')
print(faces)

image_with_detections = np.copy(image)

RED = (255, 0, 0)
LINE_THICKNESS = 5

for (x, y, w, h) in faces:
    cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), RED, LINE_THICKNESS)

plt.figure(figsize=(20, 10))
plt.imshow(image_with_detections)
plt.show()
