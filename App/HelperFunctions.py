from typing import Tuple

import numpy as np


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
