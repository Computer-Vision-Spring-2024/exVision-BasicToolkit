import matplotlib.colors as mcolors
import numpy as np
from PyQt5.QtCore import pyqtSignal

from Classes.EffectsWidgets.EdgeDetectorGroupBox import EdgeDetectorGroupBox
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton


class EdgeDetector(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, *args, **kwargs):
        super(EdgeDetector, self).__init__(parent)
        # For naming the instances of the effect
        EdgeDetector._instance_counter += 1
        self.title = f"Edge Detector.{EdgeDetector._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title
        # ----------------------------------------------------------------------- Attributes --------------------------------------------------------------------
        self.lookup = {
            "sobel_3x3": self.sobel_3x3,
            "sobel_5x5": self.sobel_5x5,
            "roberts": self.roberts,
            "prewitt": self.prewitt,
            "laplacian": self.laplacian,
            "canny": self.canny,
        }
        self.current_working_image = None  # this represents the current image on which we will perform all operations (MUST BE GRAYSCALE)
        self.current_detector_type = "sobel_3x3"  # default detector 
        self.edged_image = None  # output image 

        self.edge_widget = EdgeDetectorGroupBox(self.title)
        # Pass the NoiseGroupBox instance to the Noise class
        self.edge_widget.edge_effect = self

        self.edge_widget.edge_widget_combo_box.currentTextChanged.connect(
            self.update_detector_type
        )

        # Store the attributes of the effect to be easily stored in the images instances.
        self.attributes = self.attributes_dictionary()

    # ---------------------------------------------------------------------------- Setters --------------------------------------------------------------------------
    def attributes_dictionary(self):
        """
        Description:
            - Returns a dictionary containing the attributes of the effect.
        """
        return {
            # "edge detector type": self.current_edge_detector,
            "edge detector type": self.current_detector_type,
            "groupbox": self.edge_widget,
        }

    def update_detector_type(self):
        """ This method update the current detector type and apply the new detector on the current working image. """
        self.current_detector_type = (
            self.edge_widget.edge_widget_combo_box.currentText()
        )
        self.apply_detector()
        self.attributes_updated.emit(self.edged_image) # emitting a signal of new edged image catptured, so that it gets displayed instead of the current one.

    def set_working_image(self, image):
        """
        Descripion:
            set the current image that the edge detector object deals with. 
        Parameters:
        - image: numpy.ndarray
            The input image.
        """
        self.current_working_image = image # Always grayscale  
    # ----------------------------------------------------------------------------- Methods ------------------------------------------------------------------------
    def to_grayscale(self, image):
        """
        Descripion:
            - Convert an image to grayscale by averaging the red, green, and blue channels for each pixel.

        Parameters:
        - image: numpy.ndarray
            The input image.

        Returns:
        - numpy.ndarray
            The grayscale image.
        """
        grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return grayscale_image

    def apply_detector(self):
        """   
        Descripion:
            - This is the master method, that calls the different detector, based on the current detector type selected.
        Returns:
            - numpy.ndarray
                rgb image or grayscale image depending on the detector type. 
         """
        output_image = self.lookup[self.current_detector_type](
            self.current_working_image # lookup dict returns a method corresponding to the selected detector type, then the current working image is passed as parameter. 
        ) # this trick has been introduced to avoid code repetition or branching.

        if len(output_image) == 2:
            output_image = self.get_directed_image(
                output_image
            )  # if the output contains directionality list, then include it in output image. 

        self.edged_image = output_image

        return self.edged_image

    def get_directed_image(self, image):
        """
        Descripion:
            - return rgb image that contians directionality info.

        Parameters:
        - image (tuple of numpy.ndarray ): the first array contains magnitude of edges, the second contains the gradient direction.

        Returns:
        - numpy.ndarray
            rgb image.
        """
        mag = image[0]
        direction = image[1]
        direction_normalized = direction / np.pi 
        image_shape = image[0].shape
        hue = direction_normalized * 360 
        hsv_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        hsv_image[..., 0] = hue.astype(np.uint8)
        hsv_image[..., 1] = mag.astype(np.uint8)
        hsv_image[..., 2] = mag.astype(np.uint8)
        rgb_image = mcolors.hsv_to_rgb(hsv_image / 255)
        return rgb_image

    def padding_image(self, image, width, height, pad_size):
        """
        Descripion:
            - Pad the input image with zeros from the four direction with the specified padding size.

        Parameters:
            - image (numpy.ndarray): The input image.
            - width (int): The desired width of the padded image.
            - height (int): The desired height of the padded image.
            - pad_size (int): The size of padding to add around the image.

        Returns:
            numpy.ndarray: The padded image.

        """
        padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size)) # zeros matrix 
        padded_image[pad_size:-pad_size, pad_size:-pad_size] = image  
        return padded_image

    def convolve_2d(self, image, kernel, multiply=True):
        """
        Description: 
            - Perform 2D convolution on the input image with the given kernel.

        Parameters:
            - image (numpy.ndarray): The input image.
            - kernel (numpy.ndarray): The convolution kernel.
            - multiply (bool, optional): Whether to multiply the kernel with the image region. 
                                    Defaults to True.

        Returns:
            numpy.ndarray: The convolved image.

        """
        image_height, image_width = image.shape
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2

        if pad_size == 0: # if the kernel size is 1 x 1 
            padded_image = image 
            normalize_value = 2 # the value by which we normalize the convolved region of image.
        else:
            # padding the image to retain image size. 
            normalize_value = kernel_size * kernel_size 
            padded_image = self.padding_image(
                image, image_width, image_height, pad_size
            )

        output_image = np.zeros_like(image) #  output image size == input image size

        for i in range(image_height):
            for j in range(image_width):
                neighborhood = padded_image[i : i + kernel_size, j : j + kernel_size ] # slice out the region

                # optimization trick (usage case maybe for average filter, where it's useless to multiply the kernel with the region)
                if multiply:
                    output_image[i, j] = np.sum(neighborhood * kernel)
                else:
                    output_image[i, j] = np.sum(neighborhood) * (1 / normalize_value)
            
        return np.clip(output_image, 0, 255)

    def get_edges_with_gradient_direction(self, image, x_kernel, y_kernel, rotated_coord=False):
        """
        Description:
            - Compute edges and gradient directions of the input image using the specified x and y kernels.

        Parameters:
            - image (numpy.ndarray): The input image.
            - x_kernel (numpy.ndarray): The kernel for computing the x component of the gradient.
            - y_kernel (numpy.ndarray): The kernel for computing the y component of the gradient.
            - Rotated_coord (bool, optional): Whether to rotate the gradient directions by 45 degrees. Defaults to False.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The resultant edges image.
                - The gradient directions image.

        """
        x_component = self.convolve_2d(image, x_kernel)
        y_component = self.convolve_2d(image, y_kernel)
        resultant = np.abs(x_component) + abs(y_component) 
        resultant = resultant / np.max(resultant) * 255 # image displayed 
        direction = np.arctan2(y_component, x_component) 
        if rotated_coord: # in case of roberts detector which compute gradient diagonal-wise
            direction = (direction + np.pi / 4) % np.pi  # Wrap angles back to [0, π]
        return (resultant, direction)

    def sobel_3x3(self, image):
        """
        Apply the Sobel 3x3 edge detection filter to the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The resultant edges image.
                - The gradient directions image.
        """
        dI_dX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        dI_dY = np.rot90(dI_dX)
        return self.get_edges_with_gradient_direction(image, dI_dX, dI_dY)

    def sobel_5x5(self, image):
        """
        Apply the Sobel 5x5 edge detection filter to the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The resultant edges image.
                - The gradient directions image.

        """
        dI_dX = np.array(
            [
                [-1, -2, 0, 2, 1],
                [-2, -3, 0, 3, 2],
                [-3, -5, 0, 5, 3],
                [-1, -2, 0, 2, 1],
                [-2, -3, 0, 3, 2],
            ]
        )
        dI_dY = np.rot90(dI_dX)
        return self.get_edges_with_gradient_direction(image, dI_dX, dI_dY)

    def roberts(self, image):
        """
        Apply the Roberts edge detection filter to the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The resultant edges image.
                - The gradient directions image.

        """
        secondary_diag = np.array([[0, 1], [-1, 0]])
        main_diag = np.rot90(secondary_diag)
        return self.get_edges_with_gradient_direction(image, secondary_diag, main_diag, rotated_coord=True)  

    def prewitt(self, image):
        """
        Apply the Prewitt edge detection filter to the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The resultant edges image.
                - The gradient directions image.

        """
        dI_dX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        dI_dY = np.rot90(dI_dX)
        return self.get_edges_with_gradient_direction(image, dI_dX, dI_dY)

    def laplacian(self, image):
        """
        Apply the Laplacian edge detection filter to the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The resultant edges image.

        """
        kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]])
        edged_image = self.convolve_2d(image, kernel, True)
        return edged_image

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        """
        Suppress non-maximum gradients to thin edges in the input image.

        Parameters:
            gradient_magnitude (numpy.ndarray): The gradient magnitude image.
            gradient_direction (numpy.ndarray): The gradient direction image.

        Returns:
            numpy.ndarray: The image after non-maximum suppression.
        """
        image_height, image_width = gradient_magnitude.shape
        suppressed_image = np.zeros((image_height, image_width), dtype=np.uint8)

        angles_degrees = gradient_direction * 180 / np.pi
        angles_degrees[angles_degrees < 0] += 180 

        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                neighbor_intensity1, neighbor_intensity2 = 255, 255

                if (0 <= angles_degrees[i, j] < 22.5) or (157.5 <= angles_degrees[i, j] <= 180):
                    neighbor_intensity2 = gradient_magnitude[i, j - 1]
                    neighbor_intensity1 = gradient_magnitude[i, j + 1]

                elif 22.5 <= angles_degrees[i, j] < 67.5:
                    neighbor_intensity2 = gradient_magnitude[i - 1, j + 1]
                    neighbor_intensity1 = gradient_magnitude[i + 1, j - 1]

                elif 67.5 <= angles_degrees[i, j] < 112.5:
                    neighbor_intensity2 = gradient_magnitude[i - 1, j]
                    neighbor_intensity1 = gradient_magnitude[i + 1, j]

                elif 112.5 <= angles_degrees[i, j] < 157.5:
                    neighbor_intensity2 = gradient_magnitude[i + 1, j + 1]
                    neighbor_intensity1 = gradient_magnitude[i - 1, j - 1]

                if (gradient_magnitude[i, j] >= neighbor_intensity1) and (gradient_magnitude[i, j] >= neighbor_intensity2):
                    suppressed_image[i, j] = gradient_magnitude[i, j]  # Keeping
                else:
                    suppressed_image[i, j] = 0

        return suppressed_image

    def threshold(self, img, lowThresholdRatio=0.05, highThresholdRatio=0.2):
        """
        Apply thresholding to the input image to identify potential edge pixels.

        Parameters:
            img (numpy.ndarray): The input image.
            lowThresholdRatio (float, optional): The ratio of the high threshold to use as the low threshold. 
                                                Defaults to 0.05.
            highThresholdRatio (float, optional): The ratio of the maximum intensity to use as the high threshold. 
                                                Defaults to 0.2.

        Returns:
            Tuple[numpy.ndarray, int, int]: A tuple containing:
                - The thresholded image.
                - The intensity value for weak edge pixels.
                - The intensity value for strong edge pixels.
        """
        highThreshold = img.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio 

        image_height, image_width = img.shape
        res = np.zeros((image_height, image_width), dtype=np.int32)

        weak = np.int32(25) 
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)  # left zeros

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res, weak, strong)

    def hysteresis(self, img, weak, strong=255):
        """
        Apply hysteresis to strengthen weak edge pixels that are connected to strong edge pixels.

        Parameters:
            img (numpy.ndarray): The input image containing edge pixels.
            weak (int): The intensity value for weak edge pixels.
            strong (int, optional): The intensity value for strong edge pixels. Defaults to 255.

        Returns:
            numpy.ndarray: The image after applying hysteresis.
        """
        image_height, image_width = img.shape

        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                if (
                    img[i, j] == weak
                ):  # these weak edges are considered to be strong, if they are connected to strong edges
                    if (
                        (img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img

    def canny(self, image):
        """
        Apply the Canny edge detection algorithm to the input image.

        The Canny edge detection algorithm consists of the following steps:
        1. Applying Filter to smooth out the edges. (optional)
        2. Compute gradient magnitudes and directions using the Sobel operator.
        3. Suppress non-maximum gradients to thin edges.
        4. Apply double thresholding to identify potential edge pixels.
        5. Use hysteresis to trace edges and discard weak edge pixels.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The output image with detected edges.

        """
        edged_image = self.sobel_3x3(image) # [magnitude, direction]
        suppressed_image = self.non_maximum_suppression(edged_image[0], edged_image[1])
        thresholded_image_info = self.threshold(suppressed_image)
        output_image = self.hysteresis(*thresholded_image_info)
        return output_image
