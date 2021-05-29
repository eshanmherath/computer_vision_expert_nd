"""
what makes filters high and low-pass; why is a Sobel filter high-pass and a Gaussian filter low-pass?

Well, you can actually visualize the frequencies that these filters block out by taking a look at their
fourier transforms. The frequency components of any image can be displayed after doing a Fourier Transform (FT).
An FT looks at the components of an image (edges that are high-frequency, and areas of smooth color as low-frequency),
and plots the frequencies that occur as points in spectrum. So, let's treat our filters as small images, and display
them in the frequency domain!

Areas of white or light gray, allow that part of the frequency spectrum through!
Areas of black mean that part of the spectrum is blocked out of the image.

Recall that the low frequencies in the frequency spectrum are at the center of the frequency transform image,
and high frequencies are at the edges. You should see that the Gaussian filter allows only low-pass frequencies through,
which is the center of the frequency transformed image.
The sobel filters block out frequencies of a certain orientation
and a laplace (all edge, regardless of orientation) filter, should block out low-frequencies!


"""

import matplotlib.pyplot as plt
import numpy as np

# Define gaussian, sobel, and laplacian (edge) filters

gaussian = (1 / 9) * np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# laplacian, edge filter
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

filters = [gaussian, sobel_x, sobel_y, laplacian]
filter_name = ['gaussian', 'sobel_x', 'sobel_y', 'laplacian']

# perform a fast fourier transform on each filter
# and create a scaled, frequency transform image
f_filters = [np.fft.fft2(x) for x in filters]
f_shift = [np.fft.fftshift(y) for y in f_filters]
frequency_tx = [np.log(np.abs(z) + 1) for z in f_shift]

# display 4 filters
for i in range(len(filters)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(frequency_tx[i], cmap='gray')
    plt.title(filter_name[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
