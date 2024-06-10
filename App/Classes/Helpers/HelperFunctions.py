import random
from typing import *

import numpy as np
from Classes.Helpers.Features import *
from PIL import Image
from skimage.transform import resize


def is_grayscale(img):
    """
    Determine if an image is grayscale or not.

    Parameters:
    image (np.ndarray): The input image.

    Returns:
    int: Returns 1 if the image is grayscale, 0 if the image is colored.
    """

    if len(img.shape) == 2:
        return 1
    else:
        return 0


def merge_channels(channel1, channel2, channel3):
    """
    Merge three single-channel images into a single three-channel image.

    Parameters:
    channel1, channel2, channel3 (np.ndarray): Single-channel images to be merged.

    Returns:
    np.ndarray: The output three-channel image.
    """
    channel1 = np.squeeze(channel1)

    # Check if all input images have the same shape
    assert (
        channel1.shape == channel2.shape == channel3.shape
    ), "All input images must have the same shape."

    # Create an empty array with the same width and height as the input images, but with a depth of 3
    merged_image = np.empty((channel1.shape[0], channel1.shape[1], 3))

    # Assign each channel to the corresponding layer of the output image
    merged_image[..., 0] = channel1
    merged_image[..., 1] = channel2
    merged_image[..., 2] = channel3

    return merged_image


def mean_std_dev(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the mean and standard deviation of an image.

    Parameters:
    image (np.ndarray): The input image. It should be a numpy array.

    Returns:
    Tuple[float, float]: A tuple containing the mean and standard deviation of the image.
    """

    # Calculate the mean
    mean = np.mean(image)

    # Calculate the standard deviation
    std_dev = np.std(image)

    return mean, std_dev


def min_max(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the minimum and maximum pixel values of an image.

    Parameters:
    image (np.ndarray): The input grayscale image. It should be a numpy array.

    Returns:
    Tuple[float, float]: A tuple containing the minimum and maximum pixel values of the image.
    """

    # Calculate the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    return min_val, max_val


import numpy as np


def _3d_colored_or_not(image: np.ndarray) -> bool:
    """
    Determine if an image is colored or grayscale.

    This function checks if the input image is colored or grayscale.
    It does this by comparing the RGB channels of the image. If all channels are equal, it means the image is grayscale.
    If not, the image is colored.

    Parameters:
    image (np.ndarray): The input image. It should be a 3D numpy array with shape (height, width, 3) for colored images,
                        or a 2D array with shape (height, width) for grayscale images.

    Returns:
    int: Returns 0 if the image is grayscale, 1 if the image is colored.
    """

    # Check if the image is 2D (grayscale)
    if len(image.shape) == 2:
        return 0  # The image is grayscale

    # Check if all color channels are equal (indicating a grayscale image)
    elif np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
        image[:, :, 1], image[:, :, 2]
    ):
        return 0  # The image is grayscale
    else:
        return 1  # The image is colored


def Histogram_computation(Image: np.ndarray):
    """
    Descripion:
        - Compute the histogram of an image for each color channel separately.

    Returns:
    - Histogram: numpy.ndarray
        if Image is a 3D colored image: Histogram is a 2D array representing the histogram of the input image.
        Each row corresponds to a pixel intensity (0 to 255),
        and each column corresponds to a color channel (0 for red, 1 for green, 2 for blue).

        If Image is a 3D or 2D Grey image: Histogram is a 1D array representing the same as above.
    """
    image_height = Image.shape[0]
    image_width = Image.shape[1]
    # Check if the image is grey, but still a 3d ndarray
    _3d_color = _3d_colored_or_not(Image)
    if not _3d_color:
        # The image is a 2d ndarray, and supposed to be a grey image
        image_channels = 1
    else:
        image_channels = Image.shape[2]
    # Initialize the histogram array with zeros. The array has 256 rows, each corresponding to a pixel intensity value (0 to 255), and
    # Image_Channels columns, each corresponding to a color channel (0 for red, 1 for green, 2 for blue). Each element in the array will store the count of pixels with a specific intensity
    # value in a specific color channel.
    Histogram = np.zeros([256, image_channels])

    # Compute the histogram for each pixel in each channel
    for x in range(0, image_height):
        for y in range(0, image_width):
            # Increment the count of pixels in the histogram for the same pixel intensity at position (x, y) in the image for the current color channel (c).
            # This operation updates the histogram to track the number of pixels with a specific intensity value in each color channel separately.
            # Image[x, y, c] => gets the intensity of the pixel at that position of the image which corresponds to row number of histogram.
            # c => is the color channel which corresponds to the column number of the histogram,
            if image_channels == 1:
                Histogram[Image[x, y], 0] += 1
            else:
                for c in range(image_channels):
                    Histogram[Image[x, y, c], c] += 1

    return Histogram.astype(int)


def BGR2LAB(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an image from the BGR color space to the Lab color space.

    Parameters:
    image_array (np.ndarray): The input image in BGR color space.

    Returns:
    np.ndarray: The output image in Lab color space.
    """

    # Convert BGR to XYZ
    # First, normalize the pixel values to the range [0, 1]
    image_array = image_array.astype(np.float32) / 255

    # Apply gamma correction
    mask = image_array > 0.04045
    image_array[mask] = ((image_array[mask] + 0.055) / 1.055) ** 2.4
    image_array[~mask] = image_array[~mask] / 12.92

    # Convert to XYZ color space
    image_array = image_array * 100
    X = (
        image_array[..., 2] * 0.412453
        + image_array[..., 1] * 0.357580
        + image_array[..., 0] * 0.180423
    )
    Y = (
        image_array[..., 2] * 0.212671
        + image_array[..., 1] * 0.715160
        + image_array[..., 0] * 0.072169
    )
    Z = (
        image_array[..., 2] * 0.019334
        + image_array[..., 1] * 0.119193
        + image_array[..., 0] * 0.950227
    )

    # Normalize for D65 white point
    X = X / 95.04560
    Y = Y / 100.0000
    Z = Z / 108.8754

    # Convert XYZ to Lab
    mask = X > 0.008856
    X[mask] = X[mask] ** (1 / 3)
    X[~mask] = (7.787 * X[~mask]) + (16 / 116)

    mask = Y > 0.008856
    Y[mask] = Y[mask] ** (1 / 3)
    Y[~mask] = (7.787 * Y[~mask]) + (16 / 116)

    mask = Z > 0.008856
    Z[mask] = Z[mask] ** (1 / 3)
    Z[~mask] = (7.787 * Z[~mask]) + (16 / 116)

    # Calculate L, a, b values
    L = (116 * Y) - 16
    a = 500 * (X - Y)
    b = 200 * (Y - Z)

    # Scale the values like OpenCV
    L = L * 255 / 100
    a = a + 128
    b = b + 128

    # Stack L, a, b channels to create the final Lab image
    lab_image = np.dstack([L, a, b]).astype(np.uint8)

    return lab_image


def cumulative_summation(_1d_hist: np.ndarray) -> np.ndarray:
    """
    Description:
        - Compute the cumulative sum of a 1D array.

    Parameters:
        - _2d_hist (np.ndarray): The input 1D frequency distribution array.

    Returns:
        - np.ndarray: The output 1D array representing the cumulative summation.
    """

    # Initialize a zero array with the same shape as the input
    _1d_cdf = np.zeros(_1d_hist.shape)

    # Set the first element of the cumulative sum array to be the first element of the input array
    _1d_cdf[0] = _1d_hist[0]

    # Iterate over the input array, adding each element to the cumulative sum
    for ind in range(1, _1d_hist.shape[0]):
        _1d_cdf[ind] = _1d_cdf[ind - 1] + _1d_hist[ind]

    return _1d_cdf


## ============== Helpers for tasks 3,4,5 ============== ##


# Conver to LUV colorspace
def map_rgb_luv(self, image):
    image = anti_aliasing_resize(image)
    normalized_image = (image - image.min()) / (
        image.max() - image.min()
    )  # nomalize before
    xyz_image = rgb_to_xyz(normalized_image)
    luv_image = xyz_to_luv(xyz_image)
    luv_image_normalized = (luv_image - luv_image.min()) / (
        luv_image.max() - luv_image.min()
    )  # normalize after  (point of question !!)
    # scaled_image = scale_luv_8_bits(luv_image)
    return luv_image_normalized


# Detection
WINDOW_SIZE = 15

# size and location NamedTuple objects
Size = NamedTuple("Size", [("height", int), ("width", int)])
Location = NamedTuple("Location", [("top", int), ("left", int)])


def convert_to_grey(img_RGB: np.ndarray) -> np.ndarray:
    if len(img_RGB.shape) == 3:
        grey = np.dot(img_RGB[..., :3], [0.2989, 0.5870, 0.1140])
        return grey.astype(np.uint8)
    else:
        return img_RGB.astype(np.uint8)


def convert_BGR_to_RGB(img_BGR_nd_arr: np.ndarray) -> np.ndarray:
    img_RGB_nd_arr = img_BGR_nd_arr[..., ::-1]
    return img_RGB_nd_arr


def rgb_to_xyz(rgb):
    """Convert RGB color values to XYZ color values."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    X = 0.412453 * R + 0.35758 * G + 0.180423 * B
    Y = 0.212671 * R + 0.71516 * G + 0.072169 * B
    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    return np.stack((X, Y, Z), axis=-1)


def xyz_to_luv(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    constant = 903.3
    un = 0.19793943
    vn = 0.46832096

    epsilon = 1e-12  # to prevent division by zero
    u_prime = 4 * X / (X + 15 * Y + 3 * Z + epsilon)
    v_prime = 9 * Y / (X + 15 * Y + 3 * Z + epsilon)

    L = np.where(Y > 0.008856, 116 * Y ** (1 / 3) - 16, constant * Y)
    U = 13 * L * (u_prime - un)
    V = 13 * L * (v_prime - vn)

    return np.stack((L, U, V), axis=-1)


def scale_luv_8_bits(luv_image):
    L, U, V = luv_image[..., 0], luv_image[..., 1], luv_image[..., 2]

    scaled_L = L * (255 / 100)
    scaled_U = (U + 134) * (255 / 354)
    scaled_V = (V + 140) * (255 / 262)

    return np.stack((L, U, V), axis=-1)


def anti_aliasing_resize(img):
    """This function can be used for resizing images of huge size to optimize the segmentation algorithm"""
    ratio = min(1, np.sqrt((512 * 512) / np.prod(img.shape[:2])))
    newshape = list(map(lambda d: int(round(d * ratio)), img.shape[:2]))
    img = resize(img, newshape, anti_aliasing=True)
    return img


def padding_matrix(matrix, width, height, pad_size):
    """
    Description:
        - Pad the input matrix with zeros from the four direction with the specified padding size.

    Parameters:
        - matrix (numpy.ndarray): The input matrix.
        - width (int): The desired width of the padded matrix.
        - height (int): The desired height of the padded matrix.
        - pad_size (int): The size of padding to add around the matrix.

    Returns:
        - numpy.ndarray: The padded matrix.
    """
    padded_matrix = np.zeros(
        (height + 2 * pad_size, width + 2 * pad_size)
    )  # zeros matrix
    padded_matrix[pad_size : pad_size + height, pad_size : pad_size + width] = matrix
    return padded_matrix


def convolve2d_optimized(input_matrix, convolution_kernel, mode="same"):
    """
    Perform a 2D convolution of an input matrix with a convolution kernel.

    Parameters:
        - input_matrix (numpy.ndarray): The input matrix to be convolved.
        - convolution_kernel (numpy.ndarray): The kernel used for the convolution.
        - mode (str): The mode of convolution, can be 'same' (default), 'valid', or 'full'.

    Returns:
        - output_matrix (numpy.ndarray): The result of the convolution.
    """

    # Get dimensions of input matrix and kernel
    input_height, input_width = input_matrix.shape
    kernel_size = convolution_kernel.shape[0]
    padding_size = kernel_size // 2

    # Pad the input matrix
    padded_matrix = padding_matrix(
        input_matrix, input_width, input_height, pad_size=padding_size
    )

    # Create an array of offsets for convolution
    offset_array = np.arange(-padding_size, padding_size + 1)

    # Create a meshgrid of indices for convolution
    x_indices, y_indices = np.meshgrid(offset_array, offset_array, indexing="ij")

    # Add the meshgrid indices to an array of the original indices
    i_indices = (
        np.arange(padding_size, input_height + padding_size)[:, None, None]
        + x_indices.flatten()
    )
    j_indices = (
        np.arange(padding_size, input_width + padding_size)[None, :, None]
        + y_indices.flatten()
    )

    # Use advanced indexing to get the regions for convolution
    convolution_regions = padded_matrix[i_indices, j_indices].reshape(
        input_height, input_width, kernel_size, kernel_size
    )

    # Compute the convolution by multiplying the regions with the kernel and summing the results
    output_matrix = np.sum(convolution_regions * convolution_kernel, axis=(2, 3))

    return output_matrix


def gaussian_weight(distance, sigma):
    """Introduce guassian weighting based on the distance from the mean"""
    return np.exp(-(distance**2) / (2 * sigma**2))


def generate_random_color():
    """
    Description:
        -   Generate a random color for the seeds and their corresponding region in the region-growing segmentation.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def _pad_image(kernel_size, grayscale_image):
    """
    Description:
        - Pads the grayscale image with zeros.

    Returns:
        - [numpy.ndarray]: A padded grayscale image.
    """
    pad_width = kernel_size // 2
    return np.pad(
        grayscale_image,
        ((pad_width, pad_width), (pad_width, pad_width)),
        mode="edge",
    )


def Normalized_histogram_computation(Image):
    """
    Compute the normalized histogram of a grayscale image.

    Parameters:
    - Image: numpy.ndarray.

    Returns:
    - Histogram: numpy array
        A 1D array representing the normalized histogram of the input image.
        It has 256 element, each element corresponds to the probability of certain pixel intensity (0 to 255).
    """
    # Get the dimensions of the image
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]
    # Initialize the histogram array with zeros. The array has 256 element, each corresponding to a pixel intensity value (0 to 255)
    Histogram = np.zeros([256])
    print(Histogram.shape)
    # Compute the histogram for each pixel in each channel
    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            # Increment the count of pixels in the histogram for the same pixel intensity at position (x, y) in the image.
            # This operation updates the histogram to track the number of pixels with a specific intensity value.
            Histogram[int(Image[x, y])] = Histogram[int(Image[x, y])] + 1
    # Normalize the histogram by dividing each bin count by the total number of pixels in the image
    Histogram /= Image_Height * Image_Width
    return Histogram


def resize_image_object(img, target_size):
    thumbnail_image = img.copy()
    thumbnail_image.thumbnail(
        target_size, Image.Resampling.LANCZOS
    )  # anti-alising-resize
    return thumbnail_image


def to_float_array(img):
    return np.array(img).astype(np.float32) / 255.0  # float division


def to_image(arr):
    return Image.fromarray(np.uint8(arr * 255.0))


def gamma(channel, coeff=2.2):
    return channel ** (1.0 / coeff)


def gleam_converion(img):
    return np.sum(gamma(img), axis=2) / img.shape[2]  # divide by 3


def integrate_image(img):
    """The padding compensates for the loss that might happen in differentiation"""
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)  # 2d integral
    return np.pad(integral, (1, 1), "constant", constant_values=(0, 0))[:-1, :-1]


def possible_combinations(size, window_size=WINDOW_SIZE):
    return range(0, window_size - size + 1)  # size can be height or width


def possible_locations(base_size: Size, window_size=WINDOW_SIZE):
    return (
        Location(left=x, top=y)
        for x in possible_combinations(base_size.width, window_size)
        for y in possible_combinations(base_size.height, window_size)
    )


def possible_feature_shapes(base_size: Size, window_size=WINDOW_SIZE):
    base_height = base_size.height
    base_width = base_size.width
    return (
        Size(height=height, width=width)
        for width in range(base_width, window_size + 1, base_width)
        for height in range(base_height, window_size + 1, base_height)
    )


# this is helper types
ThresholdPolarity = NamedTuple(
    "ThresholdPolarity", [("threshold", float), ("polarity", float)]
)

ClassifierResult = NamedTuple(
    "ClassifierResult",
    [
        ("threshold", float),
        ("polarity", int),
        ("classification_error", float),
        ("classifier", Callable[[np.ndarray], float]),
    ],
)

WeakClassifier = NamedTuple(
    "WeakClassifier",
    [
        ("threshold", float),
        ("polarity", int),
        ("alpha", float),
        ("classifier", Callable[[np.ndarray], float]),
    ],
)


def weak_classifier(
    window: np.ndarray, feature: Feature, polarity: float, theta: float
):
    return (np.sign((polarity * theta) - (polarity * feature(window))) + 1) // 2
    # computational optimization


def run_weak_classifier(window: np.ndarray, weak_classier: WeakClassifier):
    return weak_classifier(
        window,
        weak_classier.classifier,
        weak_classier.polarity,
        weak_classier.threshold,
    )


def strong_classifier(window: np.ndarray, weak_classifiers: List[WeakClassifier]):
    sum_hypotheses = 0.0
    sum_alpha = 0.0
    for cl in weak_classifiers:
        sum_hypotheses += cl.alpha * run_weak_classifier(window, cl)
        sum_alpha += cl.alpha
    vote = 1 if (sum_hypotheses >= 0.5 * sum_alpha) else 0
    how_strong = sum_hypotheses - 0.5 * sum_alpha
    return (vote, how_strong)


def normalize(im):
    return (im - im.mean()) / im.std()
