import os
from collections import defaultdict

import cv2
import numpy as np
from Classes.Effects.CornerDetection import CornerDetection
from Classes.Effects.EdgeDetector import EdgeDetector
from Classes.Effects.Equalizer import Equalizer
from Classes.Effects.FaceDetection import FaceDetection
from Classes.Effects.FaceRecognition import FaceRecognitionGroupBox
from Classes.Effects.Filter import Filter
from Classes.Effects.FreqFilters import FreqFilters
from Classes.Effects.HoughTransform import HoughTransform
from Classes.Effects.Hybrid import HybridImages
from Classes.Effects.Noise import Noise
from Classes.Effects.Normalize import Normalizer
from Classes.Effects.Segmentation import Segmentation
from Classes.Effects.SIFT import SIFT
from Classes.Effects.Snake import SNAKE
from Classes.Effects.Thresholding import Thresholding
from Classes.ExtendedWidgets.CanvasWidget import CanvasWidget
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from Classes.Helpers.HelperFunctions import (
    BGR2LAB,
    Histogram_computation,
    _3d_colored_or_not,
    cumulative_summation,
    is_grayscale,
)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QGroupBox, QMessageBox

from Classes.Effects.AdvancedThresholding import AdvancedThresholding


class Image:
    all_images = []

    def __init__(self, file_path):
        self.file_path = file_path

        self.img_data = None
        self.grayscale_img = None

        self.output_img = None  # The output image of the last appied effect
        self.cumulative_output = (
            None  # The output image of appliying all the effects as "chain"
        )

        self.applied_effects = (
            {}
        )  # Dictionary to store the applied effects and its parameters.
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

    def set_cumulative_output(self, output_data):
        """
        Description:
            - Sets the cumulative output image for the current image.
        Args:
            - output_data: The cumulative output image data to be saved.
        """
        self.cumulative_output = output_data

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
    def get_grayscale_img(self):
        return self.grayscale_img


class Backend:
    def __init__(self, ui):
        self.ui = ui

        # Variables
        ## === Image Data === ##
        self.current_image = None  # It holds the current image instance
        self.current_image_data = None  # It holds the current image data
        self.grayscale_image = None  # It holds the grayscale image data
        self.is_color = False  # It holds the state of the image (grayscale or not)
        self.output_image = None  # It holds the output image data
        self.cumulative_output = None  # It holds the cumulative output image data
        self.image_history = {}  # It holds all images that was opened by the user
        # The key is the name of the file and the value is the img data

        ## === Image Types === ##
        self.image_picker_types = "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)"
        self.allowed_types = [".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"]
        self.prohibited_types = [
            "ARRIRAW",
            "Blackmagic RAW",
            "DNG",
            "DPX",
            "EXR",
            "PSD",
            "TIFF",
        ]  # Floating-Point Images

        ## === Effects Library === ##
        self.effects_library = [
            {
                "name": "Convert to Grayscale",
                "icon": "Resources/Icons/Effects/grayscale.png",
                "function": self.display_grayscale,
            },
            {
                "name": "Add Noise",
                "icon": "Resources/Icons/Effects/noise.png",
                "function": self.add_noise,
            },
            {
                "name": "Filter Noise",
                "icon": "Resources/Icons/Effects/filter.png",
                "function": self.filter_image,
            },
            {
                "name": "Detect Edges",
                "icon": "Resources/Icons/Effects/edges.png",
                "function": self.detect_edges,
            },
            {
                "name": "Histograms and Distribution Curve",
                "icon": "Resources/Icons/Effects/histogram.png",
                "function": self.plot_histogram_and_CDF_in_new_tab,
            },
            {
                "name": "Equalizer",
                "icon": "Resources/Icons/Effects/equalizer.png",
                "function": self.equalizer,
            },
            {
                "name": "Normalize Image",
                "icon": "Resources/Icons/Effects/normalize.png",
                "function": self.normalize_image,
            },
            {
                "name": "Local && Global Thresholding",
                "icon": "Resources/Icons/Effects/threshold.png",
                "function": self.local_and_global_thresholding,
            },
            {
                "name": "Frequency domain Filters",
                "icon": "Resources/Icons/Effects/freq_filter.png",
                "function": self.frequency_domain_filters,
            },
            {
                "name": "Hybrid Images",
                "icon": "Resources/Icons/Effects/mixer-hybrid.png",
                "function": self.hybrid_images,
            },
            {
                "name": "Active Contour (Snake)",
                "icon": "Resources/Icons/Effects/snake.png",
                "function": self.snake,
            },
            {
                "name": "Boundary Detection (Hough Transform)",
                "icon": "Resources/Icons/Effects/hough.png",
                "function": self.hough,
            },
            {
                "name": "Corner Detection",
                "icon": "Resources/Icons/Effects/corner-detection.png",
                "function": self.corner_detection,
            },
            {
                "name": "SIFT",
                "icon": "Resources/Icons/Effects/sift.png",
                "function": self.sift,
            },
            {
                "name": "Advanced Thresholding",
                "icon": "Resources/Icons/Effects/thresholding02.png",
                "function": self.thresholding,
            },
            {
                "name": "Segmentation",
                "icon": "Resources/Icons/Effects/segmentation.png",
                "function": self.segmentation,
            },
            {
                "name": "Face Recognition",
                "icon": "Resources/Icons/Effects/face_recognition.png",
                "function": self.face_recognition,
            },
            {
                "name": "Face Detection",
                "icon": "Resources/Icons/Effects/face_detection.png",
                "function": self.face_detection,
            },
        ]
        ###  End  Effects Library ###

        self.init_ui_connections()

    def init_ui_connections(self):
        """
        Description:
            - Connects the UI objects to their respective methods.
        """
        # UI Objects to Methods Connections
        ## === IMPORTING === ##
        self.ui.image_workspace.imgDropped.connect(self.load_image)
        self.ui.actionImport_Image.triggered.connect(lambda: self.load_image(None))

        ## === Images Library === ##
        self.ui.actionClassic.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionClassic.text()
            )
        )
        self.ui.actionOld_Classic.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionOld_Classic.text()
            )
        )
        self.ui.actionMedical.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionMedical.text()
            )
        )
        self.ui.actionSun_and_Plants.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionSun_and_Plants.text()
            )
        )
        self.ui.actionHigh_Resolution.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionHigh_Resolution.text()
            )
        )
        self.ui.actionFingerprints.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionFingerprints.text()
            )
        )
        self.ui.actionTextures.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionTextures.text()
            )
        )
        self.ui.actionSpecial.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionSpecial.text()
            )
        )
        self.ui.actionAdditional.triggered.connect(
            lambda: self.load_image(
                None, "Resources/ImagesLibrary/" + self.ui.actionAdditional.text()
            )
        )
        self.ui.actionLine.triggered.connect(
            lambda: self.load_image(
                None, "Resources/HoughSamples/" + self.ui.actionLine.text()
            )
        )
        self.ui.actionCircle.triggered.connect(
            lambda: self.load_image(
                None, "Resources/HoughSamples/" + self.ui.actionCircle.text()
            )
        )
        self.ui.actionEllipse.triggered.connect(
            lambda: self.load_image(
                None, "Resources/HoughSamples/" + self.ui.actionEllipse.text()
            )
        )
        ## === SideBar: Effects === ##
        # Create and add effects to the collapsed and expanded views of the left bar
        for i in range(len(self.effects_library)):
            button = QDoubleClickPushButton()
            icon = self.effects_library[i]["icon"]
            button.setIcon(QtGui.QIcon(icon))
            button.setIconSize(QtCore.QSize(24, 24))
            button.setToolTip(self.effects_library[i]["name"])
            # button.setCursor(QCursor(Qt.PointingHandCursor))
            button.doubleClicked.connect(self.effects_library[i]["function"])
            self.ui.left_bar_collapsed_VLayout.addWidget(button)

        # Add vertical spacer to push the buttons up in the collapsed view
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.ui.left_bar_collapsed_VLayout.addItem(spacerItem)

        for i in range(len(self.effects_library)):
            effect_name = "   " + self.effects_library[i]["name"]
            button = QDoubleClickPushButton(effect_name)
            icon = self.effects_library[i]["icon"]
            button.setIcon(QtGui.QIcon(icon))
            button.setIconSize(QtCore.QSize(24, 24))
            button.doubleClicked.connect(self.effects_library[i]["function"])
            self.ui.left_bar_expanded_VLayout.addWidget(
                button, alignment=QtCore.Qt.AlignLeft
            )

        # Add the same vertical spacer to the expanded view
        self.ui.left_bar_expanded_VLayout.addItem(spacerItem)

        ## === TabWidget: Close Tabs === ##
        self.ui.image_workspace.tabCloseRequested.connect(
            lambda index: self.ui.image_workspace.removeTab(index)
        )

        ## === Change the current image from the tree widget === ##
        self.ui.img_history_tree.itemDoubleClicked.connect(self.load_old_image)

        ## === Initialize a counter to keep track of the number of hybrid images created === ##
        self.hybrid_img_counter = 0

        ## === Detection Image Path === ##
        self.detection_img_path = None

        ## === Clear Image History: Initially Disabled === ##
        self.ui.clear_history_btn.clicked.connect(self.clear_history)

        ## === File Menu: Save Actions: Initially Disabled === ##
        self.ui.actionSave_current.triggered.connect(self.save_image)
        self.ui.actionSave_as.triggered.connect(self.save_image_as)
        self.ui.actionSave_all.triggered.connect(self.save_all_images)

        # Initially disable all saving and clearing actions
        self.enable_disable_actions(False)

        ## === Help Menu: Controls === ##
        self.ui.actionControls.triggered.connect(self.show_controls)

    # ==================================== Methods ===================================== #

    # Importing and Plotting Functions #
    # ================================ #

    def load_image(self, file_path=None, folder=""):
        """
        Description:
            - Loads an image from the file system or a library.
            - Stores the image in the image history tree.

        Args:
            - file_path: The path of the image to be loaded.
            - folder: The folder to be opened when the file dialog box is opened.
        """
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Open Image",
                folder,
                self.image_picker_types,
            )

        if file_path:
            # Check if the file type is allowed
            if not file_path.endswith(tuple(self.allowed_types)) or file_path.endswith(
                tuple(self.prohibited_types)
            ):
                self.show_message(
                    "Error",
                    "This image type is not allowed. Please, choose another one.",
                    QMessageBox.Critical,
                )
                return

            self.detection_img_path = file_path

            # Instantiate an image object,that will be helpful to easily access the image data
            img = Image(file_path)
            img.img_data = cv2.imread(file_path, 1)
            if img.img_data is not None:
                tab_index = self.ui.image_workspace.currentIndex()
                tab_text = self.ui.image_workspace.tabText(tab_index)
                if tab_text == "Hybrid viewport":
                    self.display_selection_dialog(img.img_data, file_path)
                else:
                    # Convert BGR to RGB to correct plotting
                    img.img_data = cv2.cvtColor(img.img_data, cv2.COLOR_BGR2RGB)
                    img.output_img = img.img_data
                    # Store the image data in the image history list
                    file_name = file_path.split("/")[-1]
                    if file_name not in self.image_history:
                        self.image_history[file_name] = img
                        # Create a new parent item for the image
                        parent_image = QtWidgets.QTreeWidgetItem([file_name])
                        if hasattr(self, "hybrid_object"):
                            for file_name, image_data in self.image_history.items():
                                is_grayscale_hybrid = is_grayscale(
                                    image_data.output_img
                                )
                                temp_img = None
                                temp_img = self.convert_to_grayscale(
                                    is_grayscale_hybrid, temp_img, image_data.output_img
                                )
                                self.hybrid_object.append_processed_image(
                                    file_name, temp_img
                                )
                        # Add the parent item to the tree widget
                        self.ui.img_history_tree.addTopLevelItem(parent_image)

                        self.display_image(img.img_data, img.output_img)
                        # To affect the recent added image
                        self.current_image = img
                        self.current_image_data = img.img_data
                        self.is_grayscale = is_grayscale(self.current_image_data)

            else:
                self.show_message("Error", "Invalid Image", QMessageBox.Critical)

        # Now, we have something to save or delete, so enable the saving and clearing options
        self.enable_disable_actions(True)

    def load_old_image(self, item):
        """
        Description:
            - Loads an old image from the image history tree.

        Args:
            - The TreeItem that was clicked to access the image history.
        """
        # Get the name of the image to access the history, (column 0)
        file_name = item.text(0)
        # If the selected tree item is a child (an effect) display the group box of the effect
        # and remove the other group boxes
        if item.parent() is not None:
            parent = item.parent().text(0)
            # Hide all groupboxes
            self.hide_all_groupboxes()
            # Get the image object to which the effect is applied
            image_data = self.image_history.get(parent)
            # If it is not the currently displayed image, set the current_image to the image of the selected effect
            if not np.array_equal(image_data, self.current_image):
                self.set_current_data(image_data)
            # Display the groupbox of the clicked effect
            self.display_groupbox(image_data, item)
            image_data.applied_effects[file_name]["final_result"]()
        else:
            # Check if the first six characters of the item name are "Hybrid", then the selected image is hybrid image
            if file_name[:6] == "Hybrid":
                tab_index = int(file_name[-1]) + 1
                # Open the tab of the selected hybrid image
                self.ui.image_workspace.setCurrentIndex(tab_index)
                # Display the groupbox of the selected hybrid image
                self.hybrid_object.hybrid_widget.setVisible(True)
                # Expand the scroll area to sea the group box
                self.ui.toggle_effect_bar(True)

            else:
                self.hide_all_groupboxes()
                image_data = self.image_history.get(file_name)
                if item.childCount() > 0:
                    # Loop over the children and display their groupboxes
                    for iterator in range(item.childCount()):
                        child = item.child(iterator)
                        self.display_groupbox(image_data, child)
                if image_data:
                    self.set_current_data(image_data)
                    self.is_grayscale = is_grayscale(self.current_image_data)
                else:
                    self.show_message(
                        "Error",
                        "Image data not found in history.",
                        QMessageBox.critical,
                    )

    def set_current_data(self, image_data):
        """
        Description:
            - Set the selected image to be the current image.

        Args:
            - The image to be set.
        """
        self.current_image = image_data
        self.current_image_data = image_data.img_data
        self.output_image = image_data.output_img
        self.display_image(self.current_image_data, self.output_image)

    def hide_all_groupboxes(self):
        """
        Description:
            - Hide all the groupboxes.
        """
        for i in range(self.ui.scroll_area_VLayout.count()):
            layout_item = self.ui.scroll_area_VLayout.itemAt(i)
            if layout_item is not None and isinstance(layout_item.widget(), QGroupBox):
                widget_to_hide = layout_item.widget()
                widget_to_hide.setVisible(False)
        self.ui.add_vertical_spacer(self.ui.scroll_area_VLayout)

    def display_groupbox(self, image_data, effect):
        """
        Description:
            - Display the group box of the selected effect and remove the other group boxes.

        Args:
            - The image that the effect is applied to.
            - The selected effect
        """
        # Get the key of the selected effect in the dictionary of the group boxes of the image
        key = effect.text(0)
        self.ui.scroll_area_VLayout.insertWidget(
            0, image_data.applied_effects[key]["groupbox"]
        )
        widget = image_data.applied_effects[key]["groupbox"]
        widget.setVisible(True)
        # Expand the scroll area to sea the group box
        self.ui.toggle_effect_bar(True)

    def display_image(self, input_img, output_img, grey="gray", axis_disabled="off"):
        """
        Description:
            - Displays an image in the main canvas.

        Args:
            - input_img: The input image to be displayed.
            - output_img: The output image to be displayed.
        """
        # Clear the previous plot
        self.ui.main_viewport_figure_canvas.figure.clear()

        # Determine layout based on image dimensions
        height, width, _ = input_img.shape
        if (width - height) > 300:  # If width is greater than the height
            ax1 = self.ui.main_viewport_figure_canvas.figure.add_subplot(
                211
            )  # Vertical layout
            ax2 = self.ui.main_viewport_figure_canvas.figure.add_subplot(212)
        else:  # If height is significantly greater than width
            ax1 = self.ui.main_viewport_figure_canvas.figure.add_subplot(
                121
            )  # Horizontal layout
            ax2 = self.ui.main_viewport_figure_canvas.figure.add_subplot(122)

        ax1.imshow(input_img, cmap="gray")
        ax1.axis(axis_disabled)
        ax1.set_title("Input Image", color="white")
        if (output_img is None): 
            ax2.imshow(np.zeros_like(input_img))
            
        else:
            ax2.imshow(output_img, cmap=grey)
        ax2.axis(axis_disabled)
        ax2.set_title("Output Image", color="white")
        # Reduce the white margins
        self.ui.main_viewport_figure_canvas.figure.subplots_adjust(
            left=0, right=1, bottom=0.05, top=0.95
        )

        # Redraw the canvas
        self.ui.main_viewport_figure_canvas.draw()

    def update_output_image(self, new_output):
        """
        Description:
            - Updates the display of the output image.
        """
        self.output_image = new_output
        self.current_image.set_output_image(new_output)
        self.display_image(self.current_image_data, new_output)

    # General Helper Method: to avoid code repetition #
    # =============================================== #
    def enable_disable_actions(self, state):
        """
        Description:
            - Enable or disable the saving and clearing actions
            depending on there is a loaded image or not.

        Args:
            - state: Boolean value, True or False
        """
        self.ui.actionSave_current.setEnabled(state)
        self.ui.actionSave_as.setEnabled(state)
        self.ui.actionSave_all.setEnabled(state)
        self.ui.clear_history_btn.setEnabled(state)

    @staticmethod
    def show_message(title, message, icon_type):
        """
        Description:
            - Shows a message dialog box with the specified icon.

        Args:
            - title: Title of the dialog box.
            - message: Message to be displayed.
            - icon_type: Type of icon to be displayed (QMessageBox.Information, QMessageBox.Critical, etc.).
        """
        msg = QMessageBox()
        msg.setIconPixmap(QIcon("icon.png").pixmap(64, 64))
        msg.setIcon(icon_type)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

    # @staticmethod
    def display_selection_dialog(self, image, path):
        """
        Description:
            - Shows a message dialog box to determine which subplot the user want to display the selected image in.

        Args:
            - image: The image to be displayed.
            - path: The path of the image.
        """
        msgBox = QMessageBox()
        msgBox.setIconPixmap(QIcon("icon.png").pixmap(64, 64))
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Select an Image")
        msgBox.setWindowTitle("Image Selection")
        msgBox.setMinimumWidth(150)

        # Set custom button text
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.button(QMessageBox.Yes).setText("Image 1")
        msgBox.button(QMessageBox.No).setText("Image 2")

        # Executing the message box
        response = msgBox.exec()
        if response == QMessageBox.Rejected:
            return
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_grayscale_hybrid = is_grayscale(img)
            img = self.convert_to_grayscale(is_grayscale_hybrid, img, img)
            if response == QMessageBox.Yes:
                self.hybrid_object.set_image(img, 1, path)
            elif response == QMessageBox.No:
                self.hybrid_object.set_image(img, 0, path)

    def _check_conversion(self):
        if self.is_grayscale == 0:
            reply = QMessageBox.question(
                None,
                "Message",
                "These effect should be applied on grayscale image.\nSo your image will be converted to grayscale.\nDo you wish to proceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.grayscale_image = self.convert_to_grayscale(
                    self.is_grayscale, self.grayscale_image, self.current_image_data
                )
                self.is_grayscale = 1
                return 1
            else:
                return 0

    # =========================== SideBar Functionalities ============================== #

    # SideBar functions that instantiate the effects #
    # ============================================== #
    def display_grayscale(self):
        """
        Descripion:
            - Convert an image to grayscale and display the image after conversion.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        self.grayscale_image = self.convert_to_grayscale(
            self.is_grayscale, self.grayscale_image, self.current_image_data
        )
        self.is_grayscale = 1
        self.display_image(self.current_image_data, self.grayscale_image)

    def convert_to_grayscale(self, flag, converted_img, to_be_coneverted_img):
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
        if flag == 0:
            converted_img = np.dot(
                to_be_coneverted_img[..., :3], [0.2989, 0.5870, 0.1140]
            )
        return converted_img

    def add_noise(self):
        """
        Description:
            - Makes an instance of the Noise effect class
            and adds it to the current image.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return

        noise_effect = Noise("Uniform", 0, 0.5, self.grayscale_image)

        # Noise Effect Signal
        noise_effect.attributes_updated.connect(self.update_output_image)

        # Get the output image after applying the noise
        self.output_image = noise_effect.output_image

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(0, noise_effect.noise_groupbox)
        self.current_image.add_applied_effect(
            noise_effect.title, noise_effect.attributes
        )

        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def filter_image(self):
        """
        Description:
            - Filter the current image with either Mean, Median, or Gaussian filter.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return

        filter_effect = Filter("Mean", "3", 0, self.grayscale_image)

        # Filter Effect Signal
        filter_effect.attributes_updated.connect(self.update_output_image)

        # Get the output image after applying the Filter
        self.output_image = filter_effect.output_image

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(0, filter_effect.filter_groupbox)
        self.current_image.add_applied_effect(
            filter_effect.title, filter_effect.attributes
        )

        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def detect_edges(self):
        """
        Description:
            - detects the edges of the current image.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return

        edge_effect = EdgeDetector()  # pass mainwindow
        edge_effect.set_working_image(
            self.grayscale_image
        )  # each time you upload or even switch to new image, feed the edge detector with this image

        # Edge Detector Effect Signal
        edge_effect.attributes_updated.connect(self.update_output_image)
        self.output_image = edge_effect.apply_detector()

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(0, edge_effect.edge_widget)
        self.current_image.add_applied_effect(edge_effect.title, edge_effect.attributes)

        # self.current_image.set_output_image(self.output_image) #  i guess, you don't have to store it as you won't use it further. it's not like the noisy or filtered image on which you will keep processing
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def equalizer(self):
        self.is_color = _3d_colored_or_not(self.current_image_data)
        if self.is_color:
            img = cv2.cvtColor(self.current_image_data, cv2.COLOR_RGB2BGR)
            lab_img = BGR2LAB(img)
            l_channel = lab_img[:, :, 0]
            a_channel = lab_img[:, :, 1]
            b_channel = lab_img[:, :, 2]
            equalizer = Equalizer(l_channel)
            l_channel_equalized = equalizer.General_Histogram_Equalization()
            self.output_image = np.dstack([l_channel_equalized, a_channel, b_channel])
            self.output_image = cv2.cvtColor(self.output_image, cv2.COLOR_LAB2RGB)
            self.plot_equlaizer_histograms(l_channel_equalized, 1)
        else:
            self.grayscale_image = self.convert_to_grayscale(
                self.is_grayscale, self.grayscale_image, self.current_image_data
            )
            self.is_grayscale = 1
            equalizer = Equalizer(self.grayscale_image)
            # self.ui.scroll_area_VLayout.insertWidget(0, equalizer.equalizer_groupbox)
            self.output_image = equalizer.General_Histogram_Equalization()
            self.plot_equlaizer_histograms(self.output_image, 0)
        # Repeated parts
        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def plot_equlaizer_histograms(self, channel: np.ndarray, color: bool):
        # Main Viewport Page (First tab of the tab widget)
        new_tab = CanvasWidget()
        # TODO: Add image name or code
        new_tab.setObjectName("Histogram_Tab")
        # Add the tab to the tab widget
        self.ui.image_workspace.addTab(new_tab, "Histogram && CDF - EQ. Channel")
        # Adjusting the axis title
        # TODO: Add image name
        if color:
            name = "of L channel in CIELAB color space "
        else:
            name = "of the Grey Levels "

        hist = Histogram_computation(channel)
        channel = np.squeeze(channel)
        cdf = cumulative_summation(hist)
        cdf_max = cdf.max() + 1e10 - 5
        cdf_normalized = cdf * float(hist.max()) / cdf_max
        second_plot = cdf_normalized[np.where(cdf != 0)[0]]
        second_plot = cdf

        # Clear the previous plot
        new_tab.canvas.figure.clear()
        fig = Figure(figsize=(12, 12))

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        # Adjust the vertical spacing between subplots
        fig.subplots_adjust(hspace=0.6)

        ax1.hist(
            channel.flatten(), 256, [0, 256], color="black", rwidth=0.75, alpha=0.6
        )
        ax1.set_title(f"Equalized Histogram {name} ")
        ax2.plot(second_plot, color="red", label="Cumulative Distribution Normalized")
        ax2.set_title(f"Cumulative Distribution Normalized {name}")
        ax3.plot(
            hist,
            color="black",
            label=" Equalized Histogram and Cumulative Distribution",
        )
        ax3.plot(cdf / 255, color="red")
        ax3.set_title(
            f"Equalized Histogram and Normalized Cumulative Distribution {name}"
        )

        # Redraw the canvas
        new_tab.canvas.figure = fig
        new_tab.canvas.draw()

    def normalize_image(self):
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        self.grayscale_image = self.convert_to_grayscale(
            self.is_grayscale, self.grayscale_image, self.current_image_data
        )
        self.is_grayscale = 1
        normalizer_effect = Normalizer(self.grayscale_image, "simple rescale norm")
        normalizer_effect.attributes_updated.connect(self.update_output_image)
        self.output_image = normalizer_effect.normalized_image

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(0, normalizer_effect.normalizer_widget)

        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def plot_histogram_and_CDF_in_new_tab(self):
        # Main Viewport Page (First tab of the tab widget)
        new_tab = CanvasWidget()
        new_tab.setObjectName("Histogram_Tab")
        # Add the tab to the tab widget
        self.ui.image_workspace.addTab(new_tab, "Histogram && CDF")
        # Set the current tab index to the newly added tab
        new_tab_index = self.ui.image_workspace.count() - 1
        self.ui.image_workspace.setCurrentIndex(new_tab_index)
        # Calculate the histogram
        histogram = Histogram_computation(self.current_image_data)
        # Clear the previous plot
        new_tab.canvas.figure.clear()
        fig = Figure(figsize=(12, 10))
        ax_hist = fig.add_subplot(211)
        ax_cdf = fig.add_subplot(212)
        # TODO: repeated
        colors = [
            "red",
            "green",
            "blue",
        ]  # Red, Green, Blue as the colored image is converted to RGB -> in loading the image function
        single_channel_color = ["black"]
        if _3d_colored_or_not(self.current_image_data):
            for i, color in enumerate(colors):
                ax_hist.hist(
                    self.current_image_data[:, :, i].flatten(),
                    256,
                    [0, 256],
                    color=color,
                    rwidth=0.75,
                    alpha=0.6,
                )
                hist_color = histogram[:, i]
                cdf = cumulative_summation(hist_color)
                cdf_max = cdf.max() + 1e10 - 5
                cdf_normalized = cdf * float(histogram.max()) / cdf_max
                ax_cdf.plot(cdf_normalized, color=colors[i])
        else:
            ax_hist.hist(
                self.current_image_data.flatten(),
                256,
                [0, 256],
                color=single_channel_color[0],
                label="grey levels",
                rwidth=0.75,
            )
            cdf = cumulative_summation(histogram)
            cdf_max = (
                cdf.max() + 1e10 - 5
            )  # TO handle the case of a totally black image
            cdf_normalized = cdf * float(histogram.max()) / cdf_max
            ax_cdf.plot(cdf_normalized, color=single_channel_color[0])
        ax_hist.set_title("Histogram of Each Color Channel")
        ax_cdf.set_title("The CDF of Each Color Channel")
        ax_hist.set_xlabel("Pixel Intensity")
        ax_cdf.set_xlabel("Pixel Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_cdf.set_ylabel("Normalized CDF")
        ax_hist.legend()
        # Redraw the canvas
        new_tab.canvas.figure = fig
        new_tab.canvas.draw()
        # TODO: repeated code from nada
        # Set the current tab index to the newly added tab
        new_tab_index = self.ui.image_workspace.count() - 1
        self.ui.image_workspace.setCurrentIndex(new_tab_index)

    def local_and_global_thresholding(self):
        """
        Description:
            - Makes an instance of the Thresholding effect class and
                     it on the current image
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return

        threshold_effect = Thresholding(9, self.grayscale_image, "Local")

        # Noise Effect Signal
        threshold_effect.attributes_updated.connect(self.update_output_image)

        # Get the output image after applying the noise
        self.output_image = threshold_effect.thresholded_image

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(
            0, threshold_effect.thresholding_groupbox
        )
        self.current_image.add_applied_effect(
            threshold_effect.title, threshold_effect.attributes
        )

        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def frequency_domain_filters(self):
        """
        Description:
            - Makes an instance of the freqFilters effect class
            and adds it to the current image.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return

        freq_effect = FreqFilters(self.grayscale_image)
        # Noise Effect Signal
        freq_effect.attributes_updated.connect(self.update_output_image)
        # Get the output image after applying the noise
        self.output_image = freq_effect.output_image
        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(
            0, freq_effect.frequency_filter_groupbox
        )
        self.current_image.add_applied_effect(freq_effect.title, freq_effect.attributes)
        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def hybrid_images(self):
        """
        Description:
            - Makes an instance of the Hybrid effect class
            and add its new tab widget with all the desired subplots.
        """
        self.hybrid_object = HybridImages("Hybrid Image")
        self.ui.scroll_area_VLayout.insertWidget(0, self.hybrid_object.hybrid_widget)
        img_name = f"Hybrid Image {self.hybrid_img_counter}"
        self.hybrid_img_counter += 1
        self.hybrid_object.processed_image_library.clear()
        for file_name, image_data in self.image_history.items():
            if not (len(image_data.output_img.shape) == 2):
                is_grayscale_hybrid = is_grayscale(image_data.output_img)
                temp_img = None
                temp_img = self.convert_to_grayscale(
                    is_grayscale_hybrid, temp_img, image_data.output_img
                )
                self.hybrid_object.append_processed_image(file_name, temp_img)
            else:
                self.hybrid_object.append_processed_image(
                    file_name, image_data.output_img
                )
        # Create a new parent item for the image
        parent_image = QtWidgets.QTreeWidgetItem([img_name])
        # Add the parent item to the tree widget
        self.ui.img_history_tree.addTopLevelItem(parent_image)
        # Hybrid Viewport Page
        self.hybrid_viewport = QtWidgets.QWidget()
        self.hybrid_viewport.setObjectName("hybrid_viewport")
        # Hybrid Viewport Layout
        self.hybrid_viewport_grid_layout = QtWidgets.QGridLayout(self.hybrid_viewport)
        self.hybrid_viewport_grid_layout.setObjectName("hybrid_gridLayout_2")
        # Set the margins of the layout
        self.hybrid_viewport_grid_layout.setContentsMargins(10, 10, 10, 10)

        # Add frames to the hybrid viewport layout
        self.hybrid_viewport_grid_layout.addWidget(self.hybrid_object.frame1, 0, 0)
        self.hybrid_viewport_grid_layout.addWidget(self.hybrid_object.frame2, 1, 0)
        self.hybrid_viewport_grid_layout.addWidget(
            self.hybrid_object.hybrid_frame, 0, 1, 2, 1
        )
        # Add the tab to the tab widget
        self.ui.image_workspace.addTab(self.hybrid_viewport, "")
        # Set the current tab index to the newly added tab
        new_tab_index = self.ui.image_workspace.count() - 1
        self.ui.image_workspace.setCurrentIndex(new_tab_index)
        self.ui.image_workspace.setTabText(new_tab_index, "Hybrid viewport")

    def snake(self):
        """
        Description:
            - Makes an instance of the snake effect class
            and adds it to the current image.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self.is_grayscale == 0:
            grayscale_image = None
            grayscale_image = self.convert_to_grayscale(
                self.is_grayscale, self.grayscale_image, self.current_image_data
            )
            display_grayscale_output_flag = 0
        else:
            grayscale_image = self.grayscale_image
            display_grayscale_output_flag = 1

        snake = SNAKE(
            self.current_image_data,
            grayscale_image,
            display_grayscale_output_flag,
            self.ui,
        )
        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(0, snake.snake_groupbox)
        self.current_image.add_applied_effect(snake.title, snake.attributes)
        self.update_tree()

    def hough(self):
        # Define a function to update the edged image
        def update_edged_image(new_edged_image):
            nonlocal edged_image
            edged_image = new_edged_image
            # Update the Hough Transform with the new edged image
            hough_effect.update_images(
                self.current_image_data, self.grayscale_image, edged_image
            )
            # Update the displayed image
            self.current_image.set_output_image(self.output_image)
            self.display_image(self.current_image_data, self.output_image)

        # Check for image data
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return

        # Initialize the Hough Transform editing pack group box
        hough_collection_groupbox = QtWidgets.QGroupBox("HT Editing Pack")
        hough_collection_groupbox_vbox = QtWidgets.QVBoxLayout()
        hough_collection_groupbox.setLayout(hough_collection_groupbox_vbox)
        self.ui.scroll_area_VLayout.insertWidget(0, hough_collection_groupbox)

        # Detect edges in the filtered image using Canny
        edge_detector = EdgeDetector()
        edge_detector.set_working_image(self.grayscale_image)
        edged_image = edge_detector.apply_detector()
        hough_collection_groupbox_vbox.addWidget(edge_detector.edge_widget)

        # Connect the attributes_updated signal of the edge detector to update the edged image
        edge_detector.attributes_updated.connect(update_edged_image)

        # Apply Hough Transform
        hough_effect = HoughTransform(
            "Line", self.current_image_data, self.grayscale_image, edged_image
        )
        # Connect signals
        hough_effect.attributes_updated.connect(self.update_output_image)
        # Get the output image after applying the Hough transform
        self.output_image = hough_effect.output_image
        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(0, hough_effect.hough_groupbox)

        self.current_image.add_applied_effect(
            hough_effect.title, hough_effect.attributes
        )
        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def corner_detection(self):
        """
        Description:
            -   Makes an instance of the corner detection effect class
                and adds it to the current image.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        corner_detection_effect = CornerDetection(self.current_image_data)

        # Corner Detection Effect Signal
        corner_detection_effect.attributes_updated.connect(self.update_output_image)

        # Get the output image after applying the noise
        self.output_image = corner_detection_effect.output_image

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(
            0, corner_detection_effect.corner_detection_group_box
        )
        self.current_image.add_applied_effect(
            corner_detection_effect.title, corner_detection_effect.attributes
        )

        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def sift(self):
        pass

    def thresholding(self):
        """
        Description:
            - Makes an instance of the AdvancedThresholding effect class
            and adds it to the current image.
        """
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        if self._check_conversion() == 0:
            return
        advanced_threshold_effect = AdvancedThresholding(self.grayscale_image, self.ui)
        # Noise Effect Signal
        advanced_threshold_effect.attributes_updated.connect(self.update_output_image)
        # Get the output image after applying the noise
        self.output_image = advanced_threshold_effect.output_image
        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(
            0, advanced_threshold_effect.thresholding_groupbox
        )
        self.current_image.add_applied_effect(advanced_threshold_effect.title, advanced_threshold_effect.attributes)
        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    def segmentation(self):
        pass

    def face_recognition(self):
        pass

    def face_detection(self):
        if self.current_image_data is None:
            self.show_message(
                "Error", "Please load an image first.", QMessageBox.Critical
            )
            return

        face_detection_effect = FaceDetection(self.detection_img_path)

        # FaceDetection Effect Signal
        face_detection_effect.attributes_updated.connect(self.update_output_image)

        # Get the output image after applying the noise
        self.current_image_data = face_detection_effect.detection_original_float
        self.output_image = face_detection_effect.detection_output_image

        # UI Changes and Setters
        self.ui.scroll_area_VLayout.insertWidget(
            0, face_detection_effect.face_detection_groupbox
        )

        self.current_image.set_output_image(self.output_image)
        self.update_tree()
        self.display_image(self.current_image_data, self.output_image)

    # ======================== Control Panel Functionalities =========================== #

    # ToolBar of the image history tab widget #
    # ======================================= #
    def update_tree(self):
        """
        Description:
            - Updates the image history tree widget with images and applied effects.
        """
        self.ui.img_history_tree.clear()  # Clear the existing items in the tree widget
        if hasattr(self, "hybrid_object"):
            self.hybrid_object.processed_image_library.clear()

        # Iterate over the image history dictionary
        for file_name, image_data in self.image_history.items():
            # Create a new parent item for the image
            parent_item = QtWidgets.QTreeWidgetItem([file_name])
            if hasattr(self, "hybrid_object"):
                temp_img = cv2.cvtColor(image_data.output_img, cv2.COLOR_BGR2RGB)
                is_grayscale_hybrid = is_grayscale(image_data.output_img)
                temp_img = self.convert_to_grayscale(
                    is_grayscale_hybrid, temp_img, temp_img
                )
                self.hybrid_object.append_processed_image(file_name, temp_img)

            # Add the parent item to the tree widget
            self.ui.img_history_tree.addTopLevelItem(parent_item)

            # Check if there are applied effects for the image
            if image_data.applied_effects:
                # Iterate over the applied effects and add them as child items under the parent item
                for effect_name in image_data.applied_effects.keys():
                    child_item_text = str(effect_name)  # Convert the key to string
                    child_item = QtWidgets.QTreeWidgetItem([child_item_text])
                    parent_item.addChild(child_item)

            parent_item.setExpanded(True)  # Expand the parent item
        for i in range(self.hybrid_img_counter):
            img_name = f"Hybrid Image {i}"
            # Create a new parent item for the image
            parent_image = QtWidgets.QTreeWidgetItem([img_name])
            # Add the parent item to the tree widget
            self.ui.img_history_tree.addTopLevelItem(parent_image)

    def clear_history(self):
        """
        Description:
            - Clears the image history, removes instances of images, all effects,
            and clears the plots on the canvas.
        """
        # Clear all variables
        self.current_image = None
        self.current_image_data = None
        self.grayscale_img = None
        self.is_color = False
        self.output_image = None
        self.cumulative_output = None

        self.image_history.clear()
        Image.all_images.clear()
        # Clear the canvas
        self.ui.main_viewport_figure_canvas.figure.clear()
        self.ui.main_viewport_figure.canvas.draw()
        for i in range(self.ui.image_workspace.count() - 1, 0, -1):
            # Remove the tab from the tab widget
            tab = self.ui.image_workspace.widget(i)
            self.ui.image_workspace.removeTab(i)
            # Delete the widget associated with the tab
            tab.deleteLater()
            del tab
        self.hybrid_img_counter = 0
        # Clear the tree
        self.update_tree()

        # Remove all applied effects group boxes in the right side panel
        for i in reversed(range(self.ui.scroll_area_VLayout.count())):
            widget = self.ui.scroll_area_VLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # disable all actions since we have no images
        self.enable_disable_actions(False)

    # ==================== Added Effects Sidebar Functionalities ======================= #

    def update_effects_sidebar(self):
        """
        Description:
            - Updates the right sidebar with the applied effects of the current image.
        """
        # Remove all applied effects group boxes in the right side panel
        for i in reversed(range(self.ui.scroll_area_VLayout.count())):
            widget = self.ui.scroll_area_VLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        self.ui.add_vertical_spacer(self.ui.scroll_area_VLayout)

        # Check if there is a current image
        if self.current_image is not None:
            # Add group boxes of applied effects for the current image
            for (
                effect_name,
                effect_attributes,
            ) in self.current_image.applied_effects.items():
                # Access the effect group box directly from the effect class
                effect_groupbox = effect_attributes["effect_class"].effect_groupbox
                self.ui.scroll_area_VLayout.insertWidget(0, effect_groupbox)

    # ========================= Main Canvas Functionalities ============================ #

    def customize_canvas(self):
        pass

    # =========================== MenuBar Functionalities ============================== #

    # FileMenu: Save Images Functionalities #
    # ===================================== #

    def show_file_dialog_for_saving(self, filter):
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Image",
            "",
            filter,
        )
        return file_path

    def save_image(self):
        """
        Description:
            - Saves the current image to the file system.
        """
        file_path = self.show_file_dialog_for_saving(
            "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)"
        )

        if file_path:
            cv2.imwrite(file_path, self.output_image)

    def save_image_as(self):
        """
        Description:
            - Allows the user to save the current image with a chosen format.
        """
        file_path = self.show_file_dialog_for_saving(
            "PNG (*.png);;JPEG (*.jpg);;Bitmap (*.bmp);;PPM (*.ppm);;PGM (*.pgm)"
        )

        if file_path:
            # Determine the format based on the file extension
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()

            # Check if the chosen format is supported
            if file_extension not in [".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"]:
                self.show_message(
                    "Error",
                    "Unsupported file format. Please choose a different format.",
                    QMessageBox.Critical,
                )
                return

            # Save the image with the chosen format
            cv2.imwrite(file_path, self.output_image)

    def save_all_images(self):
        """
        Description:
            - Saves all output images of each instance image in a dictionary.
        """
        save_directory = QFileDialog.getExistingDirectory(None, "Save All Images", "")
        if save_directory:
            images_to_save = defaultdict(list)

            for image_name, image_data in self.image_history.items():
                images_to_save[image_name].append(image_data.output_img)

            for image_name, output_images in images_to_save.items():
                for i, output_image in enumerate(output_images):
                    file_path = os.path.join(
                        save_directory, f"{image_name}_output_{i+1}.png"
                    )
                    cv2.imwrite(file_path, output_image)

            self.show_message(
                "Info",
                "All images have been saved successfully.",
                QMessageBox.Information,
            )

    # HelpMenu: Controls #
    # ================== #
    def show_controls(self):
        pass
