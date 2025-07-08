import numpy as np
from PyQt5.QtCore import pyqtSignal

from Classes.CustomWidgets import QDoubleClickPushButton
from Classes.EffectsWidgets.FreqFiltersGroupBox import FreqFiltersGroupBox


class FreqFilters(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, imageData, parent=None, *args, **kwargs):
        super(FreqFilters, self).__init__(parent)
        # For naming the instances of the effect
        FreqFilters._instance_counter += 1
        self.title = f"Frequency_filter.{FreqFilters._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes
        # The group box that will contain the effect options
        self.frequency_filter_groupbox = FreqFiltersGroupBox(self.title)
        self.frequency_filter_groupbox.setVisible(False)
        self.highpass_flag = (
            self.frequency_filter_groupbox.filter_type_comb.currentIndex()
        )  # The type of filter that will be generated
        self.cutoff_freq = (
            self.frequency_filter_groupbox.cutoff_spinbox.value()
        )  # The cutoff frequency of the filter
        self.grayscale_image = imageData  # The image that the filter will be applied to
        # Calculate the default filtered image
        self.output_image = self.apply_filter(
            self.grayscale_image, self.cutoff_freq, self.highpass_flag
        )

        # Pass the FreqFilters instance to the FreqFiltersGroupBox class
        self.frequency_filter_groupbox.freq_filters_effect = self

        # Connect the signal of the group box to the update_parameters method
        self.frequency_filter_groupbox.cutoff_spinbox.valueChanged.connect(self.update)
        self.frequency_filter_groupbox.filter_type_comb.currentIndexChanged.connect(
            self.update
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
            "High-Pass_flag": self.highpass_flag,
            "Cutoff_freq": self.cutoff_freq,
            "output": self.output_image,
            "groupbox": self.frequency_filter_groupbox,
            "final_result": self.update,
        }

    # Methods
    def update(self):
        """
        Description:
            - Updates the parameters of the frequency_filter effect depending on
                the associated effect groupbox.
        """
        self.highpass_flag = (
            self.frequency_filter_groupbox.filter_type_comb.currentIndex()
        )
        self.cutoff_freq = self.frequency_filter_groupbox.cutoff_spinbox.value()
        self.output_image = self.apply_filter(
            self.grayscale_image, self.cutoff_freq, self.highpass_flag
        )
        self.attibutes = self.attributes_dictionary()
        self.attributes_updated.emit(self.output_image)

    def create_lowpass_filter(self, img, cutoff_freq):
        """
        Description:
            - Create a low pass filter.
        Parameters:
            - img: numpy.ndarray
                The image to be filtered.
            - cutoff_freq: int
                In a low pass filter, It is the radius of the unzeroed frequency circle, the circle is centered at the zero frequency
        Returns:
            - low_pass_filter: numpy.ndarray
                The created filter.
            - fourier: numpy.ndarray
                The fourier transform of the given image

        """
        # Calculate the Fourier transform and shift the zero-frequency component to the center of the spectrum
        fourier = np.fft.fftshift(np.fft.fft2(img))
        rows, columns = fourier.shape
        # Create an array of zeros of the same size as the fourier array
        low_pass_filter = np.zeros_like(fourier)
        # Set the values at the indices inside the circle to one
        for row in range(rows):
            for column in range(columns):
                # If the distance between the middle of the array and the current position is smaller than the cutoff (radius) set the value to 1
                frequency = np.sqrt((row - rows / 2) ** 2 + (column - columns / 2) ** 2)
                if frequency <= cutoff_freq:
                    low_pass_filter[row][column] = 1

        return low_pass_filter, fourier

    def create_highpass_filter(self, img, cutoff_freq):
        """
        Description:
            - Create a high pass filter.
        Parameters:
            - img: numpy.ndarray
                The image to be filtered.
            - cutoff_freq: int
                In a high pass filter, It is the radius of the zeroed frequency circle, the circle is centered at the zero frequency
        Returns:
            - high_pass_filter: numpy.ndarray
                The created filter.
            - fourier: numpy.ndarray
                The fourier transform of the given image

        """
        # Create a low pass filter
        low_pass_filter, fourier = self.create_lowpass_filter(img, cutoff_freq)
        # Minus one from the low pass filter to turn it into a high pass, all ones are converted to zeros and all zeros are converted to ones
        high_pass_filter = 1 - low_pass_filter
        return high_pass_filter, fourier

    def apply_filter(self, img, cutoff_freq, highpass_flag):
        """
        Description:
            - Filter out the low or high frequencies from the given image.
        Parameters:
            - img: numpy.ndarray
                The image to be filtered.
            - cutoff_freq: int
                In a low pass filter, It is the radius of the unzeroed frequency circle, and In a high pass filter, It is the radius of the zeroed frequency circle, the circle is centered at the middel.
            - lowpass_flag: 0 or 1
                Flag to determine if the desired filter is low pass filter or high pass filter.

        Returns:
            - filtered_image: numpy.ndarray
                The filtered image.
        """
        if not highpass_flag:
            # Create lowpass filter
            Filter, fourier_transform = self.create_lowpass_filter(img, cutoff_freq)
        else:
            # Create highpass filter
            Filter, fourier_transform = self.create_highpass_filter(img, cutoff_freq)
        # Apply the filter to the fourier transform of the image
        fourier_transform *= Filter
        # Perform inverse fourier transform to get the output filtered image
        filtered_image = np.abs(
            np.fft.ifft2(np.fft.ifftshift(fourier_transform))
        ).astype(np.uint8)
        return filtered_image
