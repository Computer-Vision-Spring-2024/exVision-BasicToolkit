import numpy as np
from Effect_Classes_and_Helpers.EffectsWidgets.ThresholdingGroupBox import (
    ThresholdingGroupBox,
)
from Effect_Classes_and_Helpers.ExtendedWidgets.DoubleClickPushButton import (
    QDoubleClickPushButton,
)
from PyQt5.QtCore import pyqtSignal


class Thresholding(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, val01, imageData, type, parent=None, *args, **kwargs):
        super(Thresholding, self).__init__(parent)

        # For naming the instances of the effect
        Thresholding._instance_counter += 1
        self.title = f"Threshold.{Thresholding._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes
        self.type = type  # the type of thresholding that will be generated.
        self.val01 = (
            val01  # this value will be either global threshold or local_block_size.
        )
        self.grayscale_image = (
            imageData  # The image that the threshold will be added to
        )
        self.shape = self.grayscale_image.shape  # The shape of the image
        self.thresholded_image = self.calculate_threshold()

        # The group box that will contain the effect options
        self.thresholding_groupbox = ThresholdingGroupBox("Thresholding Settings")
        self.thresholding_groupbox.setVisible(False)
        # Pass the Thresholding instance to the ThresholdingGroupbox class
        self.thresholding_groupbox.thresholding_effect = self

        # Connect the signal of the group box to the update_parameters method
        self.thresholding_groupbox.block_size_spinbox.valueChanged.connect(
            self.update_attributes
        )
        self.thresholding_groupbox.threshold_type_comb.currentIndexChanged.connect(
            self.update_attributes
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
            "val01": self.val01,
            "thresholded_image": self.thresholded_image,
            "groupbox": self.thresholding_groupbox,
            "final_result": self.update_attributes,
        }

    # Methods
    def update_attributes(self):
        """
        Description:
            - Updates the parameters of the thresholding effect depending on the effect widget that is associated with it.
        """
        self.val01 = self.thresholding_groupbox.block_size_spinbox.value()
        self.type = self.thresholding_groupbox.threshold_type_comb.currentText()
        self.thresholded_image = self.calculate_threshold()
        self.attibutes = self.attributes_dictionary()
        self.attributes_updated.emit(self.thresholded_image)

    def calculate_threshold(self):
        if self.type == "Local":
            return self.generate_local_threshold()
        else:
            return self.generate_global_threshold()

    def generate_local_threshold(self):
        """
        Apply local thresholding to an image.

        Parameters:
        - img: numpy.ndarray
            The input image.
        - block_size: int
            The size of the square block for dividing the image.

        Returns:
        - numpy.ndarray
            The thresholded image.
        """
        height, width = self.grayscale_image.shape
        thresholded_img = np.zeros_like(self.grayscale_image)
        if self.val01 > 0:
            if self.val01 % 2 == 0:
                self.val01 += 1

            # Divide the image into blocks and apply mean threshold to each block
            for i in range(0, height):
                for j in range(0, width):
                    # Calculate the indices for the neighborhood block around pixel (i, j)

                    # In each iteration we consider the i,j is the center vertex and we get their boundaries according to the block_size
                    # keeping in mind not to exceed the boundaries of the image, the (// 2) divide operator because we get the center vertex boundaries in each side so we need to divide
                    # block_size by 2 to get the distance in each side.
                    x_min = max(0, i - self.val01 // 2)  # Ensure x_min >= 0
                    y_min = max(0, j - self.val01 // 2)  # Ensure y_min >= 0
                    x_max = min(
                        height - 1, i + self.val01 // 2
                    )  # height - 1 is used to ensure that x_max does not exceed the bottom boundary of the image
                    # (height - 1 is the index of the last row in the image).
                    y_max = min(width - 1, j + self.val01 // 2)  # Same for width.

                    # Extract the neighborhood block from the input image

                    # In x_max+1 and y_max+1 is used to ensure that the slicing includes the pixel at the x_max and y_max indices.
                    # In Python, when you use slicing with the form start:stop, the stop index is not included in the slice.
                    block = self.grayscale_image[x_min : x_max + 1, y_min : y_max + 1]

                    # Calculate the mean intensity of the neighborhood block
                    block_thresh = np.mean(block)

                    # Threshold the center pixel based on the mean intensity of the block
                    if self.grayscale_image[i, j] >= block_thresh:
                        thresholded_img[i, j] = 255

            return thresholded_img
        else:
            return thresholded_img

    def generate_global_threshold(self):
        """
        Apply global thresholding to an image.

        Parameters:
        - img: numpy.ndarray
            The input image.
        - thresh: int
            The threshold value.

        Returns:
        - numpy.ndarray
            The thresholded image.
        """
        thresholded_img = np.zeros_like(self.grayscale_image)

        # Set all pixels in thresholded_img to 255 where the corresponding pixel values in img are greater than thresh and vice versa set it to zero.
        thresholded_img[self.grayscale_image > self.val01] = 255
        thresholded_img[self.grayscale_image <= self.val01] = 0

        return thresholded_img
