import os

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import time
from typing import *

import cv2
import numpy as np
from Helpers.Features import *
from Helpers.HelperFunctions import convert_to_grey, convolve2d_optimized


## ============== Harris & Lambda-Minus Methods ============== ##
def on_apply_detectors_clicked(self, img_RGB, operator):
    if self.harris_current_image_RGB.any():
        self.ui.horizontalSlider_corner_tab.valueChanged.connect(
            lambda value: self.on_changing_threshold(value, img_RGB, operator)
        )

        if operator == 0:
            # call the function with default parameters
            start = time.time()
            self.apply_harris_detector_vectorized(img_RGB)
            finish = time.time()
            self.ui.consumed_time_label.setText(
                f"This Operation consumed {finish-start:.3f} seconds || "
            )
            # Activate the slider and connect with apply harris function
            self.ui.horizontalSlider_corner_tab.setEnabled(True)
            self.ui.horizontalSlider_corner_tab.setMinimum(1)
            self.ui.horizontalSlider_corner_tab.setMaximum(int(10e6))
            self.ui.horizontalSlider_corner_tab.setSingleStep(10000)
            self.ui.horizontalSlider_corner_tab.setValue(10000)
            self.ui.threshold_value_label.setText(str(10000))

        elif operator == 1:
            # call the function with default parameters
            start = time.time()
            self.apply_lambda_minus_vectorized(img_RGB)
            finish = time.time()
            self.ui.consumed_time_label.setText(
                f"This Operation consumed {finish-start:.3f} seconds || "
            )
            # Activate the slider and connect with apply lambda function
            self.ui.horizontalSlider_corner_tab.setEnabled(True)
            self.ui.horizontalSlider_corner_tab.setMinimum(1)
            self.ui.horizontalSlider_corner_tab.setMaximum(10000)
            self.ui.horizontalSlider_corner_tab.setSingleStep(1)
            self.ui.horizontalSlider_corner_tab.setValue(10)

            self.ui.threshold_value_label.setText(f"{0.01}% of max eigen value")
    return


def on_changing_threshold(self, threshold, img_RGB, operator):
    output_img = img_RGB.copy()
    if operator == 0:
        if np.all(self.harris_response_operator != None):
            # Show the slider value using a label
            self.ui.threshold_value_label.setText(str(threshold))
            # Apply threshold and store detected corners
            corner_list = np.argwhere(self.harris_response_operator > threshold)
            # Create output image

            output_img[corner_list[:, 0], corner_list[:, 1]] = (
                255,
                0,
                0,
            )  # Highlight detected corners in red
            self.display_image(
                output_img,
                self.ui.harris_output_figure_canvas,
                "Harris Output Image",
                False,
            )
        elif operator == 1:
            if np.all(self.eigenvalues != None):
                # Set the value of the threshold
                value = self.ui.horizontalSlider_corner_tab.value() / 10000.0

                # Show the slider value using a label
                self.ui.threshold_value_label.setText(f"{value}% of max eigen value")
                # Apply threshold and store detected corners
                corners = np.where(self.eigenvalues > value)

                # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
                for i, j in zip(*corners):
                    cv2.circle(output_img, (j, i), 3, (0, 255, 0), -1)  # Green color
                self.display_image(
                    output_img,
                    self.ui.harris_output_figure_canvas,
                    "Lambda-Minus Output Image",
                    False,
                )

    return


def apply_harris_detector_vectorized(
    self, img_RGB, window_size=5, k=0.04, threshold=10000
):
    """
    Apply the Harris corner detection algorithm on an RGB image in a vectorized manner.

    This method detects corners within an image using the Harris corner detection algorithm. It converts the image to grayscale, computes the gradients, and then calculates the second moment matrix. The corner response is determined by the determinant and trace of this matrix, and corners are identified based on a specified threshold.

    Parameters:
    - img_RGB (numpy.ndarray): The input image in RGB format.
    - window_size (int, optional): The size of the window used to compute the sums of the second moment matrix. Defaults to 5.
    - k (float, optional): The sensitivity factor to separate corners from edges, typically between 0.04-0.06. Defaults to 0.04.
    - threshold (int, optional): The threshold above which a response is considered a corner. Defaults to 10000.

    Returns:
    - A tuple containing:
        - A list of tuples with the x-coordinate, y-coordinate, and corner response value for each detected corner.
        - The output image with detected corners highlighted in blue.

    The method modifies the input image by highlighting detected corners in blue and displays the result using the `display_image` method.
    """
    if np.all(img_RGB != None):
        # Convert image to grayscale
        gray = convert_to_grey(img_RGB)
        self.display_image(
            gray,
            self.ui.harris_input_figure_canvas,
            "Input Image",
            False,
        )
        Ix, Iy = np.gradient(gray)
        # Compute products of derivatives
        Ixx = Ix**2
        Ixy = Iy * Ix
        Iyy = Iy**2

        # Define window function
        window = np.ones((window_size, window_size))

        # Compute sums of the second moment matrix over the window
        Sxx = convolve2d_optimized(Ixx, window, mode="same")
        Sxy = convolve2d_optimized(Ixy, window, mode="same")
        Syy = convolve2d_optimized(Iyy, window, mode="same")

        # Compute determinant and trace of the second moment matrix
        det = Sxx * Syy - Sxy**2
        trace = Sxx + Syy

        # Compute corner response
        self.harris_response_operator = det - k * (trace**2)

        # Apply threshold and store detected corners
        corner_list = np.argwhere(self.harris_response_operator > threshold)
        corner_response = self.harris_response_operator[
            self.harris_response_operator > threshold
        ]

        # Create output image
        output_img = img_RGB.copy()
        output_img[corner_list[:, 0], corner_list[:, 1]] = (
            0,
            0,
            255,
        )  # Highlight detected corners in blue
        self.display_image(
            output_img,
            self.ui.harris_output_figure_canvas,
            "Harris Output Image",
            False,
        )

        return (
            list(zip(corner_list[:, 1], corner_list[:, 0], corner_response)),
            output_img,
        )


def apply_lambda_minus_vectorized(
    self, img_RGB, window_size=5, threshold_percentage=0.01
):
    """
    Apply the Lambda-Minus corner detection algorithm on an RGB image.

    This method implements a vectorized approach to identify corners within an image using the Lambda-Minus algorithm. It involves converting the image to grayscale, computing gradients, constructing the Hessian matrix, and finding eigenvalues to determine corner points based on a specified threshold.

    Parameters:
    - img_RGB (numpy.ndarray): The input image in RGB format.
    - window_size (int, optional): The size of the window used to compute the sum of Hessian matrix elements. Defaults to 5.
    - threshold_percentage (float, optional): The percentage of the maximum eigenvalue used to set the threshold for corner detection. Defaults to 0.01.

    Returns:
    - output_image (numpy.ndarray): The RGB image with detected corners marked in green.

    The method modifies the input image by drawing green circles at the detected corner points and displays the result using the `display_image` method.
    """

    # Convert image to grayscale
    gray = convert_to_grey(img_RGB)
    output_image = img_RGB.copy()
    # Compute the gradient using Sobel 5x5 operator
    K_X = np.array(
        [
            [-1, -2, 0, 2, 1],
            [-2, -3, 0, 3, 2],
            [-3, -5, 0, 5, 3],
            [-2, -3, 0, 3, 2],
            [-1, -2, 0, 2, 1],
        ]
    )

    K_Y = (
        K_X.T
    )  # The kernel for vertical edges is the transpose of the kernel for horizontal edges

    gradient_x, gradient_y = convolve2d_optimized(
        gray, K_X, mode="same"
    ), convolve2d_optimized(gray, K_Y, mode="same")
    # Compute the elements of the H matrix
    H_xx = gradient_x * gradient_x
    H_yy = gradient_y * gradient_y
    H_xy = gradient_x * gradient_y
    # Compute the sum of the elements in a neighborhood (e.g., using a Gaussian kernel)
    # Define window function
    window = np.ones((5, 5))
    H_xx_sum = convolve2d_optimized(H_xx, window, mode="same") / 25
    H_yy_sum = convolve2d_optimized(H_yy, window, mode="same") / 25
    H_xy_sum = convolve2d_optimized(H_xy, window, mode="same") / 25

    # Compute the eigenvalues
    H = np.stack([H_xx_sum, H_xy_sum, H_xy_sum, H_yy_sum], axis=-1).reshape(-1, 2, 2)
    self.eigenvalues = np.linalg.eigvalsh(H).min(axis=-1).reshape(gray.shape)

    # Threshold to find corners
    threshold = threshold_percentage * self.eigenvalues.max()
    corners = np.where(self.eigenvalues > threshold)

    # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
    for i, j in zip(*corners):
        cv2.circle(output_image, (j, i), 3, (0, 255, 0), -1)  # Green color

    self.display_image(
        output_image,
        self.ui.harris_output_figure_canvas,
        "Lambda-Minus Output Image",
        False,
    )


def clear_right_image(self):
    # Clear existing layouts before adding canvas
    for i in reversed(range(self.right_layout.count())):
        widget = self.right_layout.itemAt(i).widget()
        # Remove it from the layout list
        self.right_layout.removeWidget(widget)
        # Remove the widget from the GUI
        widget.setParent(None)
