# backend.py
import os
import random
import tkinter as tk
from itertools import combinations

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import pickle
import time
from math import cos, sin
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from Features import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image, ImageOps, ImageTk
from PyQt5 import QtGui

# imports
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
)

# from scipy.signal import convolve2d
from scipy.signal import convolve2d
from skimage.transform import rescale, resize

# from task3_ui import Ui_MainWindow
from UI import Ui_MainWindow


class BackendClass(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ### ==== HARRIS & LAMBDA-MINUS ==== ###
        self.harris_current_image_RGB = None
        self.harris_response_operator = None
        self.eigenvalues = None
        self.change_the_icon()

        # Threshold Slider(Initially Disabled)
        self.ui.horizontalSlider_corner_tab.setEnabled(False)

        # Apply Harris Button
        self.ui.apply_harris_push_button.clicked.connect(
            lambda: self.on_apply_detectors_clicked(self.harris_current_image_RGB, 0)
        )
        self.ui.apply_harris_push_button.setEnabled(False)
        # Apply Lambda Minus
        self.ui.apply_lambda_minus_push_button.clicked.connect(
            lambda: self.on_apply_detectors_clicked(self.harris_current_image_RGB, 1)
        )
        self.ui.apply_lambda_minus_push_button.setEnabled(False)

        ### ==== SIFT ==== ###
        # Images
        self.sift_target_image = None
        self.sift_template_image = None
        self.sift_output_image = None

        # Default parameters
        self.n_octaves = 4
        self.s_value = 2
        self.sigma_base = 1.6
        self.r_ratio = 10
        self.contrast_th = 0.03
        self.confusion_factor = 0.3

        # Widgets that control the SIFT parameters
        self.ui.n_octaves_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.s_value_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.sigma_base_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.r_ratio_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.contrast_th_slider.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.confusion_factor_slider.valueChanged.connect(
            self.get_new_SIFT_parameters
        )
        # Apply SIFT Button
        self.ui.apply_sift.clicked.connect(self.apply_sift)
        self.ui.apply_sift.setEnabled(False)

        ### ==== Region-Growing ==== ###
        self.rg_input = None
        self.rg_input_grayscale = None
        self.rg_output = None
        self.rg_seeds = None
        self.rg_threshold = 20
        self.rg_window_size = 3
        self.ui.window_size_spinbox.valueChanged.connect(self.update_rg_window_size)
        self.ui.region_growing_input_figure.canvas.mpl_connect(
            "button_press_event", self.rg_canvas_clicked
        )
        self.ui.region_growing_threshold_slider.valueChanged.connect(
            self.update_region_growing_threshold
        )

        # Region Growing Buttons
        self.ui.apply_region_growing.clicked.connect(self.apply_region_growing)
        self.ui.apply_region_growing.setEnabled(False)
        self.ui.reset_region_growing.clicked.connect(self.reset_region_growing)
        self.ui.reset_region_growing.setEnabled(False)

        ### ==== Agglomerative Clustering ==== ###
        self.agglo_input_image = None
        self.agglo_output_image = None
        self.agglo_number_of_clusters = 2
        self.downsampling = False
        self.agglo_scale_factor = 4
        self.distance_calculation_method = "distance between centroids"
        self.ui.distance_calculation_method_combobox.currentIndexChanged.connect(
            self.get_agglomerative_parameters
        )
        self.agglo_initial_num_of_clusters = 25
        self.ui.apply_agglomerative.setEnabled(False)
        self.ui.apply_agglomerative.clicked.connect(self.apply_agglomerative_clustering)
        self.ui.downsampling.stateChanged.connect(self.get_agglomerative_parameters)
        self.ui.agglo_scale_factor.valueChanged.connect(
            self.get_agglomerative_parameters
        )
        self.ui.initial_num_of_clusters_spinBox.valueChanged.connect(
            self.get_agglomerative_parameters
        )

        ### ==== K_Means ==== ###
        self.k_means_input = None
        self.k_means_luv_input = None
        self.k_means_output = None
        self.n_clusters = 4
        self.max_iterations = 4
        self.spatial_segmentation = False
        self.ui.spatial_segmentation_weight_spinbox.setEnabled(False)
        self.spatial_segmentation_weight = 1
        self.centroid_optimization = True
        self.k_means_LUV = False

        # K_Means Buttons
        self.ui.apply_k_means.setEnabled(False)
        self.ui.apply_k_means.clicked.connect(self.apply_k_means)
        self.ui.spatial_segmentation.stateChanged.connect(
            self.enable_spatial_segmentation
        )

        ### ==== Mean-Shift ==== ###
        self.mean_shift_input = None
        self.mean_shift_luv_input = None
        self.mean_shift_output = None
        self.mean_shift_window_size = 200
        self.mean_shift_sigma = 20
        self.mean_shift_threshold = 10
        self.mean_shift_luv = False

        # Mean-Shift Buttons
        self.ui.apply_mean_shift.setEnabled(False)
        self.ui.apply_mean_shift.clicked.connect(self.apply_mean_shift)

        ### ==== Thresholding ==== ###
        self.thresholding_grey_input = None
        self.thresholding_output = None
        self.number_of_thresholds = 2
        self.thresholding_type = "Optimal - Binary"
        self.local_or_global = "Global"
        self.otsu_step = 1
        self.separability_measure = 0
        self.global_thresholds = None
        self.ui.thresholding_comboBox.currentIndexChanged.connect(
            self.get_thresholding_parameters
        )

        # Thresholding Buttons and checkbox
        self.ui.apply_thresholding.setEnabled(False)
        self.ui.apply_thresholding.clicked.connect(self.apply_thresholding)
        self.ui.number_of_thresholds_slider.setEnabled(False)
        self.ui.number_of_thresholds_slider.valueChanged.connect(
            self.get_thresholding_parameters
        )
        self.ui.local_checkbox.stateChanged.connect(self.local_global_thresholding)
        self.ui.global_checkbox.stateChanged.connect(self.local_global_thresholding)
        self.ui.otsu_step_spinbox.setEnabled(False)

        ### ==== PCA ==== ###
        self.PCA_test_image_index = 30
        self.PCA_weights = None
        self.PCA_eigen_faces = None
        self.face_recognition_threshold = 2900
        # Configured by the user
        self.structure_number = "one"  # Dataset folder, containing subfolders named after subjects, each containing a minimum of 5 images, with extra images limited to the quantity of the smallest subject folder.
        self.dataset_dir = "Dataset"
        faces_train, faces_test = self.store_dataset_method_one(self.dataset_dir)
        self.train_pca(faces_train)
        self.test_faces_list, self.test_labels_list = self.test_faces_and_labels(
            faces_test
        )
        self.ROC_curve()
        self.PCA_test_img = self.test_faces_list[self.PCA_test_image_index]
        self.display_image(
            self.test_faces_list[self.PCA_test_image_index],
            self.ui.PCA_input_figure_canvas,
            "Query",
            True,
        )

        # Test size is 20% by default
        # PCA cumulativa variance is 90% by default

        # PCA Buttons
        self.ui.toggle.clicked.connect(self.toggle_PCA_test_image)
        self.ui.apply_PCA.clicked.connect(self.apply_PCA)

        ### ==== Detection ==== ###
        self.detection_original_image = None
        self.detection_thumbnail_image = None
        self.detection_original_float = None
        self.detection_grayscale_image = None
        self.detection_integral_image = None
        self.ui.apply_detection.setEnabled(False)
        self.features_per_window = self.get_number_of_features_per_window()
        self.detection_models = self.upload_cascade_adaboost("new_model_15_window")
        self.weak_classifiers = self.detection_models["1st"]
        self.weak_classifiers_2 = self.detection_models["2nd"]
        self.weak_classifiers_3 = self.detection_models["3rd"]
        self.last_stage_threshold = 0
        self.ui.apply_detection.clicked.connect(self.apply_face_detection)
        self.ui.last_stage_threshold_spinbox.valueChanged.connect(
            self.get_face_detection_parameters
        )
        self.last_stage_info = None
        self.detection_output_image = None

        ### ==== General ==== ###
        # Connect menu action to load_image
        self.ui.actionLoad_Image.triggered.connect(self.load_image)

    def change_the_icon(self):
        self.setWindowIcon(QtGui.QIcon("App_Icon.png"))
        self.setWindowTitle("Computer Vision - Task 03 - Team 02")

    def load_image(self):
        # clear self.r and threshold label
        self.ui.threshold_value_label.setText("")
        self.harris_response_operator = None
        self.eigenvalues = None

        # Open file dialog if file_path is not provided
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "Images",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)",
        )

        if file_path and isinstance(file_path, str):
            # Read the matrix, convert to rgb
            img = cv2.imread(file_path, 1)
            img = convert_BGR_to_RGB(img)

            current_tab = self.ui.tabWidget.currentIndex()

            if current_tab == 0:
                self.harris_current_image_RGB = img
                self.display_image(
                    self.harris_current_image_RGB,
                    self.ui.harris_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.ui.apply_harris_push_button.setEnabled(True)
                self.ui.apply_lambda_minus_push_button.setEnabled(True)
            elif current_tab == 1:
                self.display_selection_dialog(img)
                if (
                    self.sift_target_image is not None
                    and self.sift_template_image is not None
                ):
                    self.ui.apply_sift.setEnabled(True)
            elif current_tab == 3:
                self.rg_input = img
                self.rg_input_grayscale = convert_to_grey(self.rg_input)
                self.rg_output = img
                self.display_image(
                    self.rg_input,
                    self.ui.region_growing_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.display_image(
                    self.rg_output,
                    self.ui.region_growing_output_figure_canvas,
                    "Output Image",
                    False,
                )
                self.ui.apply_region_growing.setEnabled(True)
                self.ui.reset_region_growing.setEnabled(True)
                height = self.rg_input.shape[0]
                width = self.rg_input.shape[1]
                self.ui.initial_num_of_clusters_spinBox.setMaximum(height * width)
            elif current_tab == 4:
                self.agglo_input_image = img
                self.display_image(
                    self.agglo_input_image,
                    self.ui.agglomerative_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.ui.apply_agglomerative.setEnabled(True)
            elif current_tab == 5:
                self.k_means_luv_input = self.map_rgb_luv(img)
                self.k_means_input = img

                if self.ui.k_means_LUV_conversion.isChecked():
                    self.display_image(
                        self.k_means_luv_input,
                        self.ui.k_means_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                else:
                    self.display_image(
                        self.k_means_input,
                        self.ui.k_means_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                self.ui.apply_k_means.setEnabled(True)
            elif current_tab == 6:
                self.mean_shift_luv_input = self.map_rgb_luv(img)
                self.mean_shift_input = img

                if self.ui.mean_shift_LUV_conversion.isChecked():
                    self.display_image(
                        self.mean_shift_luv_input,
                        self.ui.mean_shift_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                else:
                    self.display_image(
                        self.mean_shift_input,
                        self.ui.mean_shift_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                self.ui.apply_mean_shift.setEnabled(True)
            elif current_tab == 7:
                self.thresholding_grey_input = convert_to_grey(img)
                self.ui.number_of_thresholds_slider.setEnabled(True)
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.thresholding_input_figure_canvas,
                    "Input Image",
                    True,
                )
                self.ui.apply_thresholding.setEnabled(True)
            elif current_tab == 9:
                self.PCA_test_img = convert_to_grey(img)
                self.display_image(
                    self.PCA_test_img,
                    self.ui.PCA_input_figure_canvas,
                    "Query",
                    True,
                )
                self.apply_PCA()
            elif current_tab == 11:
                self.detection_original_image = Image.open(file_path)
                self.detection_thumbnail_image = resize_image_object(
                    self.detection_original_image, (384, 288)
                )
                self.detection_original_float = to_float_array(
                    self.detection_thumbnail_image
                )
                self.detection_grayscale_image = gleam_converion(
                    self.detection_original_float
                )
                self.detection_integral_image = integrate_image(
                    self.detection_grayscale_image
                )

                self.display_image(
                    self.detection_original_float,
                    self.ui.detection_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.ui.apply_detection.setEnabled(True)

            # Deactivate the slider and disconnect from apply harris function
            self.ui.horizontalSlider_corner_tab.setEnabled(False)
            try:
                self.ui.horizontalSlider_corner_tab.valueChanged.disconnect()
            except TypeError:
                pass

    def display_image(
        self, image, canvas, title, grey, hist_or_not=False, axis_disabled="off"
    ):
        """ "
        Description:
            - Plots the given (image) in the specified (canvas)
        """
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        if not hist_or_not:
            if not grey:
                ax.imshow(image)
            elif grey:
                ax.imshow(image, cmap="gray")
        else:
            self.ui.histogram_global_thresholds_label.setText(" ")
            if grey:
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )
            else:
                image = convert_to_grey(image)
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )

        ax.axis(axis_disabled)
        ax.set_title(title)
        canvas.figure.subplots_adjust(left=0.1, right=0.90, bottom=0.08, top=0.95)
        canvas.draw()

    # @staticmethod
    def display_selection_dialog(self, image):
        """
        Description:
            - Shows a message dialog box to determine whether this is the a template or the target image in SIFT

        Args:
            - image: The image to be displayed.
        """
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Select an Image")
        msgBox.setWindowTitle("Image Selection")
        msgBox.setMinimumWidth(150)

        # Set custom button text
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.button(QMessageBox.Yes).setText("Target Image")
        msgBox.button(QMessageBox.No).setText("Template")

        # Executing the message box
        response = msgBox.exec()
        if response == QMessageBox.Rejected:
            return
        else:
            if response == QMessageBox.Yes:
                self.sift_target_image = image
                self.display_image(
                    image,
                    self.ui.input_1_figure_canvas,
                    "Target Image",
                    False,
                )
            elif response == QMessageBox.No:
                self.sift_template_image = image
                self.display_image(
                    image,
                    self.ui.input_2_figure_canvas,
                    "Template Image",
                    False,
                )


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())
