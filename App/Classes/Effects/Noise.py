import numpy as np
from Classes.EffectsWidgets.NoiseGroupBox import NoiseGroupBox
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from PyQt5.QtCore import pyqtSignal


class Noise(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, type, val01, val02, imageData, parent=None, *args, **kwargs):
        super(Noise, self).__init__(parent)

        # For naming the instances of the effect
        Noise._instance_counter += 1
        self.title = f"Noise.{Noise._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes
        self.type = type  # The type of noise that will be generated
        self.val01 = (
            val01  # The lower limit, or pepper, or the mean depending on the noise type
        )
        self.val02 = val02  # The upper limit, or salt, or the standard deviation depending on the noise type
        self.grayscale_image = imageData  # The image that the noise will be added to
        self.shape = self.grayscale_image.shape  # The shape of the grayscale image

        # Calculate the default noisy image
        self.output_image = self.calculate_noise()  # The noise that will be generated

        # The group box that will contain the effect options
        self.noise_groupbox = NoiseGroupBox(self.title)
        self.noise_groupbox.setVisible(False)
        """
        Description of the noise_groupbox:
            Associate the effect to a specific group box of multiple input methods
            to be able to update the parameters of the effect when the user changes
            one of the options from this group box.
        """
        # Pass the Noise instance to the NoiseGroupBox class
        self.noise_groupbox.noise_effect = self

        # Connect the signal of the group box to the update_parameters method
        self.noise_groupbox.lower_spinbox.valueChanged.connect(self.update_attributes)
        self.noise_groupbox.upper_spinbox.valueChanged.connect(self.update_attributes)
        self.noise_groupbox.noise_type_comb.currentIndexChanged.connect(
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
            "val02": self.val02,
            "output": self.output_image,
            "groupbox": self.noise_groupbox,
            "final_result": self.update_attributes,
        }

    # Methods
    def update_attributes(self):
        """
        Description:
            - Updates the parameters of the noise effect depending on
                the associated effect groupbox.
        """
        self.val01 = self.noise_groupbox.lower_spinbox.value() / 50
        self.val02 = self.noise_groupbox.upper_spinbox.value() / 50
        self.type = self.noise_groupbox.noise_type_comb.currentText()
        self.output_image = self.calculate_noise()
        self.attibutes = self.attributes_dictionary()
        self.attributes_updated.emit(self.output_image)

    def calculate_noise(self):
        if self.type == "Uniform":
            return self.generate_uniform_noise()
        elif self.type == "Gaussian":
            return self.generate_gaussian_noise()
        elif self.type == "Salt & Pepper":
            return self.generate_salt_pepper_noise()

    def generate_uniform_noise(self):
        """
        Description:
            - Adds uniform noise to an image.

        Returns:
            - A numpy array representing the noisy image.
        """
        uniform_noise = np.random.uniform(self.val01, self.val02, size=self.shape)
        self.output_image = np.clip(
            self.grayscale_image + 100 * uniform_noise, 0, 255
        ).astype(np.uint8)
        return self.output_image

    def generate_gaussian_noise(self):
        """
        Description:
            - Adds Gaussian noise to an image.

        Returns:
            - A numpy array representing the noisy image.
        """
        gaussian_noise = np.random.normal(
            loc=self.val01, scale=self.val02, size=self.shape
        )
        self.output_image = np.clip(
            self.grayscale_image + 100 * gaussian_noise, 0, 255
        ).astype(np.uint8)
        return self.output_image

    def generate_salt_pepper_noise(self):
        """
        Description:
            - Adds salt and pepper noise to an image.

        Returns:
            - A numpy array representing the noisy image.
        """
        noise = np.random.uniform(0, 1, size=self.grayscale_image.shape)
        self.output_image = self.grayscale_image.copy()
        self.output_image[noise < self.val01] = 0  # Black pixels (pepper)
        self.output_image[noise > 1 - self.val02] = 255  # White pixels (salt)
        return self.output_image
