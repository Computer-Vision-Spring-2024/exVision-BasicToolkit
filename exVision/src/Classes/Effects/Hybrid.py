import numpy as np

from Classes.CustomWidgets import CustomFrame, QDoubleClickPushButton
from Classes.EffectsWidgets.HybridGroupBox import HybridGroupBox


class HybridImages(QDoubleClickPushButton):
    _instance_counter = 0

    def __init__(self, parent=None, *args, **kwargs):
        super(HybridImages, self).__init__(parent)

        # For naming the instances of the effect
        HybridImages._instance_counter += 1
        self.title = f"Hybrid.{HybridImages._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Associate the effect to a specific group box of multiple input methods to update the parameters of the effect
        self.hybrid_widget = HybridGroupBox(self.title)
        # The two input images
        self.image1 = None
        self.image2 = None
        self.frame1 = CustomFrame("Image One", "frame_image1", 1)
        self.frame2 = CustomFrame("Image Two", "frame_image2", 0)
        self.hybrid_frame = CustomFrame("Hybrid Image", "frame_hybrid", 3)
        self.frame1.imgDropped.connect(self.set_image)
        self.frame2.imgDropped.connect(self.set_image)
        self.hybrid_frame.imgDropped.connect(self.set_image)
        # Dictionary of all the images that the user processed, the The key is the name of the file and the value is the processed img data
        self.processed_image_library = {}
        # Retrieve the cutoff frequencies of the filters from the spin boxes values
        self.low_pass_cutoff_freq = self.hybrid_widget.spin_low_pass.value()
        self.high_pass_cutoff_freq = self.hybrid_widget.spin_high_pass.value()
        # Update the cutoff frequencies based on the spin boxes valueChanged signals
        self.hybrid_widget.spin_low_pass.valueChanged.connect(self.update_lowpass)
        self.hybrid_widget.spin_high_pass.valueChanged.connect(self.update_highpass)
        # Update the frequency selection based on the radio buttons of the group box toggled signal
        self.hybrid_widget.radio_low_pass.toggled.connect(self.update_filtering)
        self.hybrid_widget.radio_high_pass.toggled.connect(self.update_filtering)
        self.hybrid_widget.combobox.currentIndexChanged.connect(
            self.upload_processed_image
        )
        # Pass the HybridGroupBox instance to the Hybrid class
        self.hybrid_widget.hybrid_images_effect = self

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

    def apply_filter(self, img, cutoff_freq, lowpass_flag):
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

        if lowpass_flag:
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

    def calc_manhattan_distance(self, x_point1, y_point1, x_point2, y_point2):
        """
        Description:
            - Calculate the distance between two points, to perform nearest neghibor interpolation.

        Parameters:
            - x_point1: float
                The x-coordinate of the first point.
            - y_point1: float
                The y-coordinate of the first point.
            - x_point2: float
                The x-coordinate of the second point.
            - y_point2: float
                The y-coordinate of the second point.

        Returns:
            - distance: float
                The distance between the two given points.
        """
        distance = abs(x_point1 - x_point2) + abs(y_point1 - y_point2)
        return distance

    def resize(self, img, new_height, new_width):
        """
        Description:
            - Resize the dimentions of the given image to match the new given dimentions..

        Parameters:
            - img: numpy.ndarray
                The image to be resized.
            - new_height: int
                The desired height.
            - new_width: int
                The desired width.

        Returns:
            - resized_img: numpy.ndarray
                The resized image.
        """

        # Get the current image shape
        height, width = img.shape
        # Calculate the amount of scaling needed
        height_scaling_factor = new_height / height
        width_scaling_factor = new_width / width
        # Create a new array with the desired shape and initial value of zero
        resized_img = np.zeros((new_height, new_width), dtype=np.uint8)
        # Iterate over the pixels of the new array to assign each pixel its interpolated value

        for row in range(new_height):
            for column in range(new_width):
                # Map the pixel in the new array to its location in the original img
                mapped_row_index = row / height_scaling_factor
                mapped_column_index = column / width_scaling_factor
                # Get the four closest points to the calculated mapped point
                r1, c1 = int(mapped_row_index), int(mapped_column_index)
                if c1 + 1 < width:
                    r2, c2 = r1, c1 + 1
                else:
                    r2, c2 = r1, c1
                if r1 + 1 < height:
                    r3, c3 = r1 + 1, c1
                else:
                    r3, c3 = r1, c1
                if r1 + 1 < height and c1 + 1 < width:
                    r4, c4 = r1 + 1, c1 + 1
                else:
                    r4, c4 = r1, c1

                # Initialize the minimum distance to value infinity
                min_distance = float("inf")
                # Iterate over the four points to determine the closest one to perform nearest neighbor interpolation
                for i in range(1, 5):
                    current_r, current_c = locals()[f"r{i}"], locals()[f"c{i}"]
                    # Calculate the distance between the mapped point and the current point
                    distance = self.calc_manhattan_distance(
                        mapped_row_index, mapped_column_index, current_r, current_c
                    )
                    if distance < min_distance:
                        # store the coordinates of the closest point
                        min_distance = distance
                        nearest_row = current_r
                        nearest_column = current_c
                # Assign the value of the closest neighboring pixel to the new pixel
                resized_img[row][column] = img[nearest_row, nearest_column]
        return resized_img

    def create_hybrid_img(self, img1, img2):
        """
        Description:
            - Combine the low frequency components of the first image with the high frequency components of the second image.
        Parameters:
            - img1: numpy.ndarray
                The first image.
            - img2: numpy.ndarray
                The second image.
        Returns:
            - hybrid_img: numpy.ndarray
                The resulted hybrid image.
        """
        height1, width1 = img1.shape
        height2, width2 = img2.shape
        # Resize the image to unify their size to be able to add them
        if not (height1 == height2 and width1 == width2):
            img2 = self.resize(img2, height1, width1)
        # Determine the image that the user wants to take its low frequency components based on the radio buttons of the group box
        # and assign it to image1 and the other image is assigned to image2
        if self.hybrid_widget.radio_low_pass.isChecked():
            image1 = img1
            image2 = img2
        else:
            image1 = img2
            image2 = img1
        # apply low pass filter to the first image
        low_pass_filtered_img = self.apply_filter(image1, self.low_pass_cutoff_freq, 1)
        # apply high pass filter to the second image
        high_pass_filtered_img = self.apply_filter(
            image2, self.high_pass_cutoff_freq, 0
        )
        # Add the two filtered images
        hybrid_img = low_pass_filtered_img + high_pass_filtered_img
        # If pixel value is greater than 255 modify it to 255
        # hybrid_img[hybrid_img > 255] = 255
        return hybrid_img

    def update_filtering(self):
        """
        Description:
            - Refilter the opened images based on the new selection, if both images are not None, reobtain the hybrid image.
        """
        # If the radio button is toggled, refilter the opened images based on the new selection
        if (not self.image1 is None) and (not self.image2 is None):
            if self.hybrid_widget.radio_low_pass.isChecked():
                image1 = self.apply_filter(self.image1, self.low_pass_cutoff_freq, 1)
                image2 = self.apply_filter(self.image2, self.high_pass_cutoff_freq, 0)
            else:
                image2 = self.apply_filter(self.image2, self.low_pass_cutoff_freq, 1)
                image1 = self.apply_filter(self.image1, self.high_pass_cutoff_freq, 0)
            hybrid = self.create_hybrid_img(self.image1, self.image2)
            # Emit the modified images
            self.frame1.Display_image(image1)
            self.frame2.Display_image(image2)
            self.hybrid_frame.Display_image(hybrid)
        elif not self.image1 is None:
            if self.hybrid_widget.radio_low_pass.isChecked():
                image1 = self.apply_filter(self.image1, self.low_pass_cutoff_freq, 1)
                self.frame1.Display_image(image1)
            else:
                image1 = self.apply_filter(self.image1, self.high_pass_cutoff_freq, 0)
                self.frame1.Display_image(image1)

        elif not self.image2 is None:
            if self.hybrid_widget.radio_low_pass.isChecked():
                image2 = self.apply_filter(self.image2, self.high_pass_cutoff_freq, 0)
                self.frame2.Display_image(image2)
            else:
                image2 = self.apply_filter(self.image2, self.low_pass_cutoff_freq, 1)
                self.frame2.Display_image(image2)

    def update_lowpass(self):
        """
        Description:
            - If the lowpass cutoff frequency spin box value is changed, refilter the lowpass filtered image, if Both images were opened, reobtain the hybrid image.
        """
        self.low_pass_cutoff_freq = self.hybrid_widget.spin_low_pass.value()
        self.update_filter(
            self.image1,
            self.image2,
            self.frame1,
            self.frame2,
            self.low_pass_cutoff_freq,
            1,
        )

    def update_highpass(self):
        """
        Description:
            - If the highpass cutoff frequency spin box value is changed, refilter the highpass filtered image, if Both images were opened, reobtain the hybrid image.
        """
        self.high_pass_cutoff_freq = self.hybrid_widget.spin_high_pass.value()
        self.update_filter(
            self.image2,
            self.image1,
            self.frame2,
            self.frame1,
            self.high_pass_cutoff_freq,
            0,
        )

    def update_filter(
        self,
        first_image,
        second_image,
        first_frame,
        second_frame,
        cutoff,
        flag,
    ):
        """
        Description:
            - Refilter the image whose filter cutoff frequency is modified, if Both images were opened, reobtain the hybrid image.
        Parameters:
            - first_image: numpy.ndarray
                If the modified spin box is that of the lowpass cutoff frequency then it is the upper image, if the modified spin box is that of the highpass cutoff then it is the lower image.
            - second_image: numpy.ndarray
                If the modified spin box is that of the lowpass then it is the lower image, if the modified spin box is that of the highpass then it is the upper image.
            - first_frame
                If the modified spin box is that of the lowpass, and the radio button which says that
                the upper image is the lowpass filtered image is checked then it is self.frame1 object
                in which the lowpass refiltered image will be displayed, otherwise it is self.frame2 object
            - second_emitted_img
                If the modified spin box is that of the highpass, and the radio button which says that
                the upper image is the lowpass filtered image is checked then it is self.frame2 object
                in which the lowpass refiltered image will be displayed, otherwise it is self.frame1 object
            - cutoff_freq: int
                In a low pass filter, It is the radius of the unzeroed frequency circle, and In a high pass filter, It is the radius of the zeroed frequency circle, the circle is centered at the middel.
            - flag: 0 or 1
                Flag to determine if the desired filter is low pass filter or high pass filter.
        """
        if self.hybrid_widget.radio_low_pass.isChecked():
            if not first_image is None:
                image = self.apply_filter(first_image, cutoff, flag)
                first_frame.Display_image(image)
        else:
            if not second_image is None:
                image = self.apply_filter(second_image, cutoff, flag)
                second_frame.Display_image(image)
        # If both images are not None, reobtain the hybrid image
        if (not self.image1 is None) and (not self.image2 is None):
            hybrid = self.create_hybrid_img(self.image1, self.image2)
            self.hybrid_frame.Display_image(hybrid)

    def set_image(self, img, image1_flag, path):
        """
        Description:
            - If the highpass cutoff frequency spin box value is changed, refilter the highpass filtered image, if Both images were opened, reobtain the hybrid image.
        Parameters:
            - img: numpy.ndarray
                The browsed image.
            - image1_flag: 0 or 1
                To determine the image that should be displayed in the upper canvas or the lower one.
            - path= str
                The path of the browsed image to be displayed in the line edit
        """
        # If the browsed image is the first image (upper image)
        if image1_flag:
            # Set image data to the first image
            self.image1 = img
            # Set the value of the line edit of the first image in the group box to the path of the image
            self.hybrid_widget.line_edit1.setText(path)
        else:
            self.image2 = img
            self.hybrid_widget.line_edit2.setText(path)
        # Filter the new added image
        self.update_filtering()

    def upload_processed_image(self):
        """
        Description:
            - If the User chose a processed image from the combobox of any of the two images, display the chosen image in its canvas.
        """
        path = self.hybrid_widget.combobox.currentText()
        image = self.processed_image_library[path]
        self.set_image(image, self.hybrid_widget.checkbox_img1.isChecked(), path)
        self.hybrid_widget.combobox.currentIndexChanged.disconnect(
            self.upload_processed_image
        )
        self.hybrid_widget.combobox.setCurrentIndex(-1)
        self.hybrid_widget.combobox.currentIndexChanged.connect(
            self.upload_processed_image
        )

    def append_processed_image(self, path, image):
        """
        Description:
            - If the user processed new image, add it to the processed_image_library dictionary and add its path to the two comboboxes.
        Parameters:
            - path= str
                The path of the new processed image
            - img: numpy.ndarray
                The new processed image.
        """
        # If image path is not in the library append it
        if path not in self.processed_image_library.keys():
            self.processed_image_library[path] = image
            self.update_combobox_items()

    def update_combobox_items(self):
        """
        Description:
            - Modify the items of the two combobox based on the last modified processed images.
        """

        self.hybrid_widget.combobox.currentIndexChanged.disconnect(
            self.upload_processed_image
        )
        self.hybrid_widget.combobox.clear()
        key = list(self.processed_image_library.keys())
        for iterator in range(len(key)):
            self.hybrid_widget.combobox.addItem(key[iterator])
        self.hybrid_widget.combobox.setCurrentIndex(-1)
        self.hybrid_widget.combobox.currentIndexChanged.connect(
            self.upload_processed_image
        )
