import numpy as np
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from Classes.Helpers.HelperFunctions import Histogram_computation, cumulative_summation
from PyQt5.QtCore import pyqtSignal


class Equalizer(QDoubleClickPushButton):
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, imageData, parent=None, *args, **kwargs):
        super(Equalizer, self).__init__(parent)

        self.title = "Equalizer"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes
        self.image = imageData

    # Methods

    def General_Histogram_Equalization(self):
        """
        Perform histogram equalization on a given image channel.

        Parameters:
        channel (np.ndarray): The input image channel. it's a 2D array where each element contains spicific greyscale (or L channel) value.

        Returns:
        np.ndarray: The output image channel after histogram equalization to be merged (if it's a colored image)
        with the other channels to produce the outpub contrast-enhanced image.
        """
        # Calculate histogram of the input channel
        hist = Histogram_computation(self.image)

        # Calculate cumulative distribution function (CDF) of the histogram
        cdf = cumulative_summation(hist)

        # Normalize the CDF
        cdf_normalized = cdf * hist.max() / cdf.max()

        # Mask all pixels in the CDF with '0' intensity
        cdf_masked = np.ma.masked_equal(cdf_normalized, 0)

        # Equalize the histogram by scaling the CDF
        cdf_masked = (
            (cdf_masked - cdf_masked.min())
            * 255
            / (cdf_masked.max() - cdf_masked.min())
        )

        # Fill masked pixels with '0'
        cdf = np.ma.filled(cdf_masked, 0).astype("uint8")

        # Apply the equalization to the original image channel
        # To clarify, now cdf contains the equalized values and need to be placed at the correct indices
        # Each value in the channel changes to its equivalent value from cdf
        channel_eq = cdf[self.image]
        channel_eq = np.squeeze(channel_eq)

        return channel_eq
