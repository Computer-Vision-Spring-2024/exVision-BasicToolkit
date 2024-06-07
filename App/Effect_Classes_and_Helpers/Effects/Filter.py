import numpy as np
from Effect_Classes_and_Helpers.EffectsWidgets.FilterGroupBox import FilterGroupBox
from Effect_Classes_and_Helpers.ExtendedWidgets.DoubleClickPushButton import (
    QDoubleClickPushButton,
)
from PyQt5.QtCore import pyqtSignal


class Filter(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(
        self, type, kernel_size, sigma, imageData, parent=None, *args, **kwargs
    ):
        super(Filter, self).__init__(parent)
        # For naming the instances of the effect
        Filter._instance_counter += 1
        self.title = f"Filter.{Filter._instance_counter:03d}"
        self.setText(self.title)

        # Attributes
        self.type = type
        # The spread or standard deviation of the Gaussian distribution
        self.sigma = sigma
        self.kernel_size = int(kernel_size)

        self.grayscale_image = imageData
        self.output_image = self.calculate_filter()

        self.filter_groupbox = FilterGroupBox(self.title)
        self.filter_groupbox.setVisible(True)

        # Pass the Noise instance to the NoiseGroupBox class
        self.filter_groupbox.filter_effect = self

        # Connect the signal of the group box to the update_parameters method
        self.filter_groupbox.filter_type_comb.currentIndexChanged.connect(
            self.update_attributes
        )
        self.filter_groupbox.kernel_size_comb.currentIndexChanged.connect(
            self.update_attributes
        )
        self.filter_groupbox.sigma_spinbox.valueChanged.connect(self.update_attributes)
        # Connect the signal of the combobox to update_filter_options method in FilterGroupBox
        self.filter_groupbox.filter_type_comb.currentIndexChanged.connect(
            self.filter_groupbox.update_filter_options
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
            "type": self.type,
            "sigma": self.sigma,
            "kernel_size": self.kernel_size,
            "output": self.output_image,
            "groupbox": self.filter_groupbox,
            "final_result": self.update_attributes,
        }

    # Methods
    def update_attributes(self):
        """
        Description:
            - Updates the parameters of the noise effect depending on
                the associated effect groupbox.
        """
        self.type = self.filter_groupbox.filter_type_comb.currentText()
        self.kernel_size = int(self.filter_groupbox.kernel_size_comb.currentText())
        self.sigma = self.filter_groupbox.sigma_spinbox.value()
        self.output_image = self.calculate_filter()
        self.attibutes = self.attributes_dictionary()
        self.attributes_updated.emit(self.output_image)

    def calculate_filter(self):
        if self.type == "Mean":
            return self.mean_filter()
        elif self.type == "Weighed Average":
            return self.weighed_average_filter()
        elif self.type == "Gaussian":
            return self.gaussian_filter()
        elif self.type == "Median":
            return self.median_filter()
        elif self.type == "Max":
            return self.max_filter()
        elif self.type == "Min":
            return self.min_filter()
        else:
            raise ValueError("Unexpected filter type: " + self.type)

    def _pad_image(self):
        """
        Description:
            - Pads the grayscale image with zeros.

        Returns:
            - [numpy.ndarray]: A padded grayscale image.
        """
        pad_width = self.kernel_size // 2
        return np.pad(
            self.grayscale_image,
            ((pad_width, pad_width), (pad_width, pad_width)),
            mode="edge",
        )

    def _apply_filter(self, filter_function):
        """
        Description:
            -   Applies a filter to an image.

        Args:
            -   filter_function: A function that takes a window and returns a value.
                This function is determined based on the type of filter to be applied.
                It is either mean, median or gaussian.

        Returns:
            -  [numpy ndarray]: A filtered image using a filter function.
        """
        padded_image = self._pad_image()
        filtered_image = np.zeros_like(self.grayscale_image)
        for i in range(self.grayscale_image.shape[0]):
            for j in range(self.grayscale_image.shape[1]):
                window = padded_image[
                    i : i + self.kernel_size, j : j + self.kernel_size
                ]
                filtered_image[i, j] = filter_function(window)
        return filtered_image

    def _gaussian_filter_kernel(self, kernel_size, sigma):
        """
        Description:
            - Generates a Gaussian filter kernel.

        Args:
            - kernel_size: Size of the square kernel (e.g., 3x3).
            - sigma: Standard deviation of the Gaussian distribution.

        Returns:
            - A numpy array representing the Gaussian filter kernel.
        """
        offset = kernel_size // 2

        x = np.arange(-offset, offset + 1)[:, np.newaxis]
        y = np.arange(-offset, offset + 1)
        x_squared = x**2
        y_squared = y**2

        kernel = np.exp(-(x_squared + y_squared) / (2 * sigma**2))
        kernel /= 2 * np.pi * (sigma**2)  # for normalization

        return kernel

    def mean_filter(self):
        """
        Description:
            -   Applies a mean filter to an image.

        Returns:
            -   [numpy ndarray]: A filtered image using a mean filter.
        """
        return self._apply_filter(np.mean)

    def weighed_average_filter(self):
        pass

    def gaussian_filter(self):
        """
        Description:
            - Applies a Gaussian filter to an image.

        Returns:
            - A numpy array representing the filtered image.
        """
        rows, cols = self.grayscale_image.shape[:2]
        kernel = self._gaussian_filter_kernel(self.kernel_size, self.sigma)
        pad_width = kernel.shape[0] // 2
        padded_image = np.pad(
            self.grayscale_image,
            ((pad_width, pad_width), (pad_width, pad_width)),
            mode="edge",
        )
        gaussian_filtered_image = np.zeros_like(self.grayscale_image)

        for i in range(rows):
            for j in range(cols):
                window = padded_image[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                gaussian_filtered_image[i, j] = np.sum(window * kernel)

        return gaussian_filtered_image

    def median_filter(self):
        """
        Description:
            -   Applies a median filter to an image.

        Returns:
            -   [numpy ndarray]: A filtered image using a median filter.
        """
        return self._apply_filter(np.median)

    def max_filter(self):
        """
        Description:
            - Applies a max filter to an image.

        Returns:
            - A numpy array representing the filtered image.
        """
        return self._apply_filter(np.max)

    def min_filter(self):
        """
        Description:
            - Applies a min filter to an image.

        Returns:
            - A numpy array representing the filtered image.
        """
        return self._apply_filter(np.min)
