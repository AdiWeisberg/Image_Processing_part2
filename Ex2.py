import math
import scipy
from math import sqrt, pi, cos, sin, atan2
from collections import defaultdict
from canny import edgeDetectionCanny
from collections import defaultdict
import numpy as np
import scipy.signal as sig
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


#1.1 Convulation1D
def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:

    kernel1 = np.flipud(kernel1)
    maxSize = max(inSignal.size, kernel1.size)

    inSignalnew = np.pad(inSignal, (math.floor(maxSize / 2) + math.floor(kernel1.size / 2), math.floor(maxSize / 2)),
                         'constant', constant_values=(0, 0))
    kernel2 = np.pad(kernel1, (math.floor(maxSize / 2), math.floor(maxSize / 2)),
                     'constant', constant_values=(0, 0))

    if inSignalnew.size < kernel2.size:
        np.append(inSignalnew, 0)
    elif inSignalnew.size > kernel2.size:
        np.append(kernel2, 0)

    maxAfter = max(inSignalnew.size, kernel2.size)
    print(inSignalnew)
    print(kernel2)
    output = np.zeros(maxSize)
    print(output)
    for i in range(0, maxSize):
        tempSum = 0
        for j in range(0, maxAfter):
            tempSum += inSignalnew[j] * kernel2[j]

#1.1 Convulation2D
def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).

    kernel = np.flipud(np.fliplr(kernel2))  # Flip the kernel
    output = np.zeros_like(inImage)  # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((inImage.shape[0] + 2, inImage.shape[1] + 2))
    image_padded[1:-1, 1:-1] = inImage
    for x in range(inImage.shape[1]):  # Loop over every pixel of the image
        for y in range(inImage.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()
    return output

# 2.1 Derivatives
def convDerivative(inImage: np.ndarray) -> np.ndarray:
    height = inImage.shape[0]
    width = inImage.shape[1]
    Hx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

    Hy = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])
    img_x = conv2D(inImage, Hx)
    img_y = conv2D(inImage, Hy)
    grad_mag = (img_x * 2 + img_y * 2) ** (0.5)

    gradientAngle = np.zeros((height, width))
    for i in range(1, height - 1, 1):
        for j in range(1, width - 1, 1):
            if img_x[i][j] == 0 and img_y[i][j] == 0:
                gradientAngle[i][j] = 0
            elif img_x[i][j] == 0 and img_y[i][j] != 0:
                gradientAngle[i][j] = 90
            else:
                x = math.degrees(math.atan(img_y[i][j] / img_x[i][j]))
                if x < 0:
                    x = 360 + x
                if x >= 170 or x < 350:
                    x = x - 180
                gradientAngle[i][j] = x
    # gradientMagnitude = (gradientMagnitude / np.max(gradientMagnitude)) * 255
    # gradientMagnitude = (gradientMagnitude / np.max(gradientMagnitude)) * 255
    cv2.imshow('mag', grad_mag)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (grad_mag)

#2.2 Blurring
def blurImage1(inImage:np.ndarray,kernelSize:np.ndarray)->np.ndarray:
    return conv2D(inImage, gaussian_kernel(kernelSize))

def blurImage2(inImage:np.ndarray,kernelSize:np.ndarray)->np.ndarray:
    return cv2.filter2D(inImage,cv2.CV_64F,cv2.getGaussianKernel(kernelSize,1))

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x*2 + y2) / (2.0*sigma*2))) * normal
    print(g)
    return g

# 3 Edge detection
def edgeDetectionSobel(I:np.ndarray)->(np.ndarray,np.ndarray):
    m, img_x, img_y = convDerivative(I)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    f_x = cv2.filter2D(img_x, cv2.CV_64F, sobel_x)
    f_y = cv2.filter2D(img_y, cv2.CV_64F, sobel_y)
    # (1/8)((f_x * 2 + f_y * 2) * (0.5))
    return (1/8)*f_x, (1/8)*f_y


def edgeDetectionZeroCrossingSimple(I:np.ndarray)->(np.ndarray,np.ndarray):
    pass
def edgeDetectionZeroCrossingLOG(I:np.ndarray)->(np.ndarray,np.ndarray):
    pass

# 3.4 edge-Detection-Canny
def edgeDetectionCanny(I:np.ndarray)->(np.ndarray,np.ndarray):
    input_pixels = I.load()
    width = I.width
    height = I.height
    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)
    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)
    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)
    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)
    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)
    output_image = Image.new("RGB", I.size)
    draw = ImageDraw.Draw(output_image)
    #save image
    for x, y in keep:
        draw.point((x, y), (255, 255, 255))
    output_image.save("canny.png")

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep
    return list(keep)

# 4 Hough Circles
def houghCircle(I:np.ndarray,minRadius:float,maxRadius:float)-> np.ndarray:
    # Find circles
    rmin = minRadius
    rmax = maxRadius
    steps = 100
    threshold = 0.4

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x, y in edgeDetectionCanny(I):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(x, y, r)
            circles.append((x, y, r))

    return circles


# Example run for convolution Derivative:
# img = cv2.imread('source_image.jpg', 0)
# img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
# cv2.imshow('image', convDerivative(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Example run for Hough circles transform:
input_image = Image.open("BBJdtSV.jpg")
circles = houghCircle(input_image, 18, 20)
# Output image:
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)
for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))
# Save output image
output_image.save("result.png")
