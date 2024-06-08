import os

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time
from typing import *

import cv2
from Classes.EffectsWidgets.CornerDetectionGroupBox import CornerDetectionGroupBox
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from Classes.Helpers.Features import *
from Classes.Helpers.HelperFunctions import convert_to_grey, convolve2d_optimized
from PyQt5.QtCore import pyqtSignal


class CornerDetection(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, imageData, parent=None, *args, **kwargs):
        super(CornerDetection, self).__init__(parent)
        # For naming the instances of the effect
        CornerDetection._instance_counter += 1
        self.title = f"Corner Detection.{CornerDetection._instance_counter:03d}"
        self.setText(self.title)

        # Attributes
        self.corener_detection_threshold = 0
        self.elapsed_time = 0
        self.input_image = imageData
        self.grayscale_image = None
        self.output_image = None

        self.corner_detection_group_box = CornerDetectionGroupBox(self.title)
        self.corner_detection_group_box.setVisible(True)

        # Pass the CornerDetection instance to the CornerDetectionGroupBox class
        self.corner_detection_group_box.corner_detection_effect = self

        # Connect the apply buttons of the effect
        self.corner_detection_group_box.apply_harris.clicked.connect(
            lambda: self.on_apply_detectors_clicked(self.input_image, 0)
        )
        self.corner_detection_group_box.apply_lambda_minus.clicked.connect(
            lambda: self.on_apply_detectors_clicked(self.input_image, 1)
        )

        # Store the attributes of the effect to be easily stored in the images instances.
        self.attributes = self.attributes_dictionary()

    # Setters
    def attributes_dictionary(self):
        """
        Description:
            - Returns a dictionary containing the attributes of the effect.
        """
        return {
            "threshold": self.corener_detection_threshold,
            "elapsed time": self.elapsed_time,
            "output": self.output_image,
            "groupbox": self.corner_detection_group_box,
        }

    # Methods
    def update_attributes(self):
        """
        Description:
            - Updates the parameters of the noise effect depending on
                the associated effect groupbox.
        """
        self.corener_detection_threshold = (
            self.corner_detection_group_box.threshold_slider.value()
        )
        # self.output_image = None
        self.attibutes = self.attributes_dictionary()
        self.attributes_updated.emit(self.output_image)

    def detect_corners(self):
        """
        Description:
            -   Connect the buttons to the detection function and get the output image and store it
        """
        # Apply the Harris corner detection algorithm
        self.apply_harris_detector_vectorized(self.input_image)

        # Display the output image with detected corners
        self.display_image(self.output_image)

    def on_apply_detectors_clicked(self, img_RGB, operator):
        if self.harris_current_image_RGB.any():
            self.corner_detection_group_box.threshold_slider.valueChanged.connect(
                lambda value: self.on_changing_threshold(value, img_RGB, operator)
            )

            if operator == 0:
                # call the function with default parameters
                start = time.time()
                self.apply_harris_detector_vectorized(img_RGB)
                finish = time.time()
                self.corner_detection_group_box.elapsed_time_label.setText(
                    f"This Operation consumed {finish-start:.3f} seconds || "
                )
                # Activate the slider and connect with apply harris function
                self.corner_detection_group_box.threshold_slider.setEnabled(True)
                self.corner_detection_group_box.threshold_slider.setMinimum(1)
                self.corner_detection_group_box.threshold_slider.setMaximum(int(10e6))
                self.corner_detection_group_box.threshold_slider.setSingleStep(10000)
                self.corner_detection_group_box.threshold_slider.setValue(10000)
                self.corner_detection_group_box.threshold_label.setText(
                    f"Threshold: {str(10000)}"
                )

            elif operator == 1:
                # call the function with default parameters
                start = time.time()
                self.apply_lambda_minus_vectorized(img_RGB)
                finish = time.time()
                self.corner_detection_group_box.elapsed_time_label.setText(
                    f"This Operation consumed {finish-start:.3f} seconds || "
                )
                # Activate the slider and connect with apply lambda function
                self.corner_detection_group_box.threshold_slider.setEnabled(True)
                self.corner_detection_group_box.threshold_slider.setMinimum(1)
                self.corner_detection_group_box.threshold_slider.setMaximum(10000)
                self.corner_detection_group_box.threshold_slider.setSingleStep(1)
                self.corner_detection_group_box.threshold_slider.setValue(10)

                self.corner_detection_group_box.threshold_label.setText(
                    f"Threshold: {0.01}% of max eigen value"
                )
        return

    def on_changing_threshold(self, threshold, img_RGB, operator):
        self.output_image = img_RGB.copy()
        if operator == 0:
            if np.all(self.harris_response_operator != None):
                # Show the slider value using a label
                self.corner_detection_group_box.threshold_label.setText(str(threshold))
                # Apply threshold and store detected corners
                corner_list = np.argwhere(self.harris_response_operator > threshold)
                # Create output image

                self.output_image[corner_list[:, 0], corner_list[:, 1]] = (
                    255,
                    0,
                    0,
                )  # Highlight detected corners in red

            elif operator == 1:
                if np.all(self.eigenvalues != None):
                    # Set the value of the threshold
                    value = (
                        self.corner_detection_group_box.threshold_slider.value()
                        / 10000.0
                    )

                    # Show the slider value using a label
                    self.corner_detection_group_box.threshold_label.setText(
                        f"{value}% of max eigen value"
                    )
                    # Apply threshold and store detected corners
                    corners = np.where(self.eigenvalues > value)

                    # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
                    for i, j in zip(*corners):
                        cv2.circle(
                            self.output_image, (j, i), 3, (0, 255, 0), -1
                        )  # Green color

        return

    def apply_harris_detector_vectorized(
        self, img_RGB, window_size=5, k=0.04, threshold=10000
    ):
        """
        Apply the Harris corner detection algorithm on an RGB image in a vectorized manner.

        This method detects corners within an image using the Harris corner detection algorithm. It converts the image to grayscale, computes the gradients, and then calculates the second moment matrix. The corner response is determined by the determinant and trace of this matrix, and corners are identified based on a specified threshold.

        Parameters:
            -   img_RGB (numpy.ndarray): The input image in RGB format.
            -   window_size (int, optional): The size of the window used to compute the sums of the second moment matrix. Defaults to 5.
            -   k (float, optional): The sensitivity factor to separate corners from edges, typically between 0.04-0.06. Defaults to 0.04.
            -   threshold (int, optional): The threshold above which a response is considered a corner. Defaults to 10000.

        Returns:
            -   A tuple containing:
            -   A list of tuples with the x-coordinate, y-coordinate, and corner response value for each detected corner.
            -   The output image with detected corners highlighted in blue.

        The method modifies the input image by highlighting detected corners in blue and displays the result using the `display_image` method.
        """
        if np.all(img_RGB != None):
            # Convert image to grayscale
            self.grayscale_image = convert_to_grey(img_RGB)
            Ix, Iy = np.gradient(self.grayscale_image)
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
            self.output_image = img_RGB.copy()
            self.output_image[corner_list[:, 0], corner_list[:, 1]] = (
                0,
                0,
                255,
            )  # Highlight detected corners in blue

            return (
                list(zip(corner_list[:, 1], corner_list[:, 0], corner_response)),
                self.output_image,
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
        self.grayscale_image = convert_to_grey(img_RGB)
        self.output_image = img_RGB.copy()
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
            self.grayscale_image, K_X, mode="same"
        ), convolve2d_optimized(self.grayscale_image, K_Y, mode="same")
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
        H = np.stack([H_xx_sum, H_xy_sum, H_xy_sum, H_yy_sum], axis=-1).reshape(
            -1, 2, 2
        )
        self.eigenvalues = (
            np.linalg.eigvalsh(H).min(axis=-1).reshape(self.grayscale_image.shape)
        )

        # Threshold to find corners
        threshold = threshold_percentage * self.eigenvalues.max()
        corners = np.where(self.eigenvalues > threshold)

        # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
        for i, j in zip(*corners):
            cv2.circle(self.output_image, (j, i), 3, (0, 255, 0), -1)  # Green color

    # def clear_right_image(self):
    #     # Clear existing layouts before adding canvas
    #     for i in reversed(range(self.right_layout.count())):
    #         widget = self.right_layout.itemAt(i).widget()
    #         # Remove it from the layout list
    #         self.right_layout.removeWidget(widget)
    #         # Remove the widget from the GUI
    #         widget.setParent(None)
