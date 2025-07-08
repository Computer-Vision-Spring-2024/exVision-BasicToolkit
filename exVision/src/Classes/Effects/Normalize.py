import numpy as np
from PyQt5.QtCore import pyqtSignal

from Classes.CustomWidgets import QDoubleClickPushButton
from Classes.EffectsWidgets.NormalizeGroupBox import NormalizeGroupBox
from Classes.Helpers.HelperFunctions import mean_std_dev, min_max


class Normalizer(QDoubleClickPushButton):
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, imageData, normalizerType, parent=None, *args, **kwargs):
        super(Normalizer, self).__init__(parent)

        self.title = "Normalizer"
        self.setText(self.title)

        # Attributes
        self.image = imageData
        self.normalizer_type = normalizerType
        self.normalized_image = self.calculate_normalized_img()

        # Alpha & Beta norm attributes
        self.alpha = 0.3
        self.beta = 0.6

        self.lookup = {
            "simple rescale norm": self.simple_rescale_norm,
            "Zero Mean && Unit Variance norm": self.Zero_Mean_and_Unit_Variance_norm,
            "Min Max Scaling norm": self.Min_Max_Scaling_norm,
            "alpha beta norm": self.alpha_beta_norm,
        }

        self.normalizer_widget = NormalizeGroupBox(self.title)
        self.normalizer_widget.setVisible(False)
        # Pass the NoiseGroupBox instance to the Noise class
        self.normalizer_widget.normalizer_effect = self

        self.normalizer_widget.normalizer_combo_box.currentTextChanged.connect(
            self.update_normalizer_type
        )
        self.normalizer_widget.alpha_spinbox.valueChanged.connect(
            self.update_normalizer_type
        )
        self.normalizer_widget.beta_spinbox.valueChanged.connect(
            self.update_normalizer_type
        )

    def update_normalizer_type(self):
        self.current_normalizer_type = (
            self.normalizer_widget.normalizer_combo_box.currentText()
        )
        if self.current_normalizer_type == "alpha beta norm":
            self.alpha = self.normalizer_widget.alpha_spinbox.value()
            self.beta = self.normalizer_widget.beta_spinbox.value()
            self.calculate_normalized_img()
        else:
            self.normalized_image = self.calculate_normalized_img()
        self.attributes_updated.emit(self.normalized_image)

    def calculate_normalized_img(self):
        if self.normalizer_type == "simple rescale norm":
            return self.simple_rescale_norm()
        elif self.normalizer_type == "Zero Mean && Unit Variance norm":
            return self.Zero_Mean_and_Unit_Variance_norm()
        elif self.normalizer_type == "Min Max Scaling norm":
            return self.Min_Max_Scaling_norm()
        elif self.normalizer_type == "alpha beta norm":
            return self.alpha_beta_norm()

    def simple_rescale_norm(self) -> np.ndarray:
        """
        Normalize a grayscale image by rescaling its pixel values to the range [0, 1].

        Parameters:
        image_array_grey_scale (np.ndarray): The input grayscale image.

        Returns:
        np.ndarray: The output image after normalization.
        """

        # Rescale the pixel values of the grayscale image to the range [0, 1]
        # by dividing each pixel value by the maximum possible value (255)
        self.normalized_image = self.image / 255.0

        return self.normalized_image

    def Zero_Mean_and_Unit_Variance_norm(self) -> np.ndarray:
        """
        Normalize an image to zero mean and unit variance.

        This function ensures that the pixel value distribution of the input grayscale image
        has a mean of 0 and a standard deviation of 1. This is often used in machine learning
        pre-processing to standardize the input data.

        Parameters:
        image_array_grey_scale (np.ndarray): The input grayscale image. It should be a numpy array.

        Returns:
        np.ndarray: The normalized image.
        """

        # Calculate the mean and standard deviation
        mean, std_dev = mean_std_dev(self.image)

        # Normalize the image to have zero mean and unit variance
        self.normalized_image = (self.image - mean) / std_dev

        return self.normalized_image

    def Min_Max_Scaling_norm(self) -> np.ndarray:
        """
        Normalize an image using Min-Max scaling.

        This function ensures that the pixel value distribution of the input grayscale image
        is scaled between 0 and 1. This is often used in machine learning pre-processing to
        standardize the input data.

        Parameters:
        image_array_grey_scale (np.ndarray): The input grayscale image. It should be a numpy array.

        Returns:
        np.ndarray: The normalized image.
        """

        # Calculate the minimum and maximum pixel values
        min_val, max_val = min_max(self.image)

        # Normalize the image to have values between 0 and 1
        self.normalized_image = (self.image - min_val) / (max_val - min_val)

        return self.normalized_image

    def alpha_beta_norm(self) -> np.ndarray:
        """
        Normalize an image to a specified range [alpha, beta].

        This function scales the pixel value distribution of the input grayscale image
        to the range [alpha, beta].

        Parameters:
        image (np.ndarray): The input grayscale image. It should be a numpy array.
        alpha (float): The lower bound of the range.
        beta (float): The upper bound of the range.

        Returns:
        np.ndarray: The normalized image.
        """

        # Check that alpha and beta are in the correct range and relation
        assert 0 <= self.alpha <= 1, "Alpha should be in the range [0, 1]"
        assert 0 <= self.beta <= 1, "Beta should be in the range [0, 1]"
        assert self.alpha < self.beta, "Alpha should be less than Beta"

        # Calculate the minimum and maximum pixel values
        min_val = np.min(self.image)
        max_val = np.max(self.image)

        # Normalize the image to the range [alpha, beta]
        self.normalized_image = self.alpha + (self.image - min_val) * (
            self.beta - self.alpha
        ) / (max_val - min_val)
        return self.normalized_image
