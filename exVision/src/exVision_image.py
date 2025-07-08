class Image:
    all_images = []

    def __init__(self, file_path):
        self.file_path = file_path

        self.img_data = None
        self.grayscale_img = None

        self.output_img = None  # The output image of the last appied effect

        self.applied_effects = {}
        # Dictionary to store the applied effects and its parameters.
        # They will be shown in the tree.
        Image.all_images.append(self)
        # To facilitate the access to the images, we will store them in a list
        # and they will be shown in the tree widget.

    # =================================== Setters ====================================== #
    def set_output_image(self, output_data):
        """
        Description:
            - Sets the output image for the current image.
        Args:
            - output_data: The output image data to be saved.
        """
        self.output_img = output_data

    def add_applied_effect(self, effect_name, effect_attributes):
        """
        Description:
            - Store the effects that were applied on the image in a dictionary
                to be shown in the tree widget.

        Args:
            - effect_name [String]: Name of the effect, the key in the dictionary.
            - effect_attributes [Dictionary]: Attributes of the effect (e.g., type, value1, value2).

        Note that these are formed inside each effect class and we are just passing it to the class to set/store it.
        """
        self.applied_effects[effect_name] = effect_attributes

    # =================================== Getters ====================================== #
    def get_original_img(self):
        return self.img_data

    def get_grayscale_img(self):
        return self.grayscale_img

    def get_output_img(self):
        return self.output_img
