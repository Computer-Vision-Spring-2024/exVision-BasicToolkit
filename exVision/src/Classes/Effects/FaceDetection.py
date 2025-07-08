import os
import pickle
from typing import *

import numpy as np
from PIL import Image
from PyQt5.QtCore import pyqtSignal

from Classes.CustomWidgets import QDoubleClickPushButton
from Classes.EffectsWidgets.FaceDetectionGroupBox import FaceDetectionGroupBox
from Classes.Helpers.Features import *
from Classes.Helpers.HelperFunctions import (
    WINDOW_SIZE,
    Location,
    Size,
    gleam_converion,
    integrate_image,
    normalize,
    possible_feature_shapes,
    possible_locations,
    resize_image_object,
    strong_classifier,
    to_float_array,
)


class FaceDetection(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, imagePath, parent=None, *args, **kwargs):
        super(FaceDetection, self).__init__(parent)

        # For naming the instances of the effect
        FaceDetection._instance_counter += 1
        self.title = f"Noise.{FaceDetection._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes
        self.detection_original_image_path = imagePath
        self.detection_original_image = None
        self.detection_thumbnail_image = None
        self.detection_original_float = None
        self.detection_grayscale_image = None
        self.detection_integral_image = None
        self.detection_output_image = None

        self.calculate_images()

        self.features_per_window = self.get_number_of_features_per_window()
        self.detection_models = self.upload_cascade_adaboost("new_model_15_window")
        self.weak_classifiers = self.detection_models["1st"]
        self.weak_classifiers_2 = self.detection_models["2nd"]
        self.weak_classifiers_3 = self.detection_models["3rd"]
        self.last_stage_threshold = 0
        self.last_stage_info = None

        # The group box that will contain the effect options
        self.face_detection_groupbox = FaceDetectionGroupBox(self.title)
        self.face_detection_groupbox.setVisible(True)

        # Pass the FaceDetection instance to the FaceDetectionGroupBox class
        self.face_detection_groupbox.face_detection_effect = self

        # Connect signals from the groupbox to the functions in this class
        self.face_detection_groupbox.last_stage_threshold_spinbox.valueChanged.connect(
            self.get_face_detection_parameters
        )
        self.face_detection_groupbox.apply_face_detection.clicked.connect(
            self.apply_face_detection
        )

    # Setters
    def calculate_images(self):
        self.detection_original_image = Image.open(self.detection_original_image_path)
        self.detection_thumbnail_image = resize_image_object(
            self.detection_original_image, (384, 288)
        )
        self.detection_original_float = to_float_array(self.detection_thumbnail_image)
        self.detection_grayscale_image = gleam_converion(self.detection_original_float)
        self.detection_integral_image = integrate_image(self.detection_grayscale_image)

    def update_attributes(self):
        self.attributes_updated.emit(self.detection_output_image)

    ## ============== Detection ============== ##
    def get_face_detection_parameters(self):
        self.last_stage_threshold = (
            self.face_detection_groupbox.last_stage_threshold_spinbox.value()
        )

    def get_number_of_features_per_window(self):
        feature2h = list(
            Feature2h(location.left, location.top, shape.width, shape.height)
            for shape in possible_feature_shapes(Size(1, 2), WINDOW_SIZE)
            for location in possible_locations(shape, WINDOW_SIZE)
        )
        # --------------------------------------------------------------
        feature2v = list(
            Feature2v(location.left, location.top, shape.width, shape.height)
            for shape in possible_feature_shapes(Size(2, 1), WINDOW_SIZE)
            for location in possible_locations(shape, WINDOW_SIZE)
        )
        # --------------------------------------------------------------
        feature3h = list(
            Feature3h(location.left, location.top, shape.width, shape.height)
            for shape in possible_feature_shapes(Size(1, 3), WINDOW_SIZE)
            for location in possible_locations(shape, WINDOW_SIZE)
        )
        # --------------------------------------------------------------
        feature3v = list(
            Feature3v(location.left, location.top, shape.width, shape.height)
            for shape in possible_feature_shapes(Size(3, 1), WINDOW_SIZE)
            for location in possible_locations(shape, WINDOW_SIZE)
        )
        # ---------------------------------------------------------------
        feature4 = list(
            Feature4(location.left, location.top, shape.width, shape.height)
            for shape in possible_feature_shapes(Size(2, 2), WINDOW_SIZE)
            for location in possible_locations(shape, WINDOW_SIZE)
        )

        features_per_window = feature2h + feature2v + feature3h + feature3v + feature4

        return features_per_window

    def upload_cascade_adaboost(self, dir):
        models = {"1st": list(), "2nd": list(), "3rd": list()}

        for filename in os.listdir(dir):
            if filename.endswith(".pickle"):  # Check if the file is a pickle file
                file_path = os.path.join(dir, filename)
                with open(file_path, "rb") as file:
                    loaded_objects = pickle.load(file)
                    models[filename[:3]].append(loaded_objects)
        return models

    def apply_face_detection(self):
        self.get_face_detection_parameters()
        rows, cols = self.detection_integral_image.shape[:2]
        HALF_WINDOW = WINDOW_SIZE // 2

        face_positions_1 = list()
        face_positions_2 = list()
        face_positions_3 = list()
        face_positions_3_strength = list()

        normalized_integral = integrate_image(
            normalize(self.detection_grayscale_image)
        )  # to reduce lighting variance

        for row in range(HALF_WINDOW + 1, rows - HALF_WINDOW):
            for col in range(HALF_WINDOW + 1, cols - HALF_WINDOW):
                curr_window = normalized_integral[
                    row - HALF_WINDOW - 1 : row + HALF_WINDOW + 1,
                    col - HALF_WINDOW - 1 : col + HALF_WINDOW + 1,
                ]

                # First cascade stage
                probably_face, _ = strong_classifier(curr_window, self.weak_classifiers)
                if probably_face < 0.5:
                    continue
                face_positions_1.append((row, col))

                probably_face, strength = strong_classifier(
                    curr_window, self.weak_classifiers_2
                )
                if probably_face < 0.5:
                    continue
                face_positions_2.append((row, col))

                probably_face, strength = strong_classifier(
                    curr_window, self.weak_classifiers_3
                )
                if probably_face < 0.5:
                    continue
                face_positions_3.append((row, col))
                face_positions_3_strength.append(strength)

        self.last_stage_info = (face_positions_3, face_positions_3_strength)
        self.truncate_candidates()

    def render_candidates(self, image: Image.Image, candidates: List[Tuple[int, int]]):
        HALF_WINDOW = WINDOW_SIZE // 2
        canvas = to_float_array(image.copy())
        for row, col in candidates:
            canvas[
                row - HALF_WINDOW - 1 : row + HALF_WINDOW, col - HALF_WINDOW - 1, :
            ] = [
                1.0,
                0.0,
                0.0,
            ]
            canvas[
                row - HALF_WINDOW - 1 : row + HALF_WINDOW, col + HALF_WINDOW - 1, :
            ] = [
                1.0,
                0.0,
                0.0,
            ]
            canvas[
                row - HALF_WINDOW - 1, col - HALF_WINDOW - 1 : col + HALF_WINDOW, :
            ] = [
                1.0,
                0.0,
                0.0,
            ]
            canvas[
                row + HALF_WINDOW - 1, col - HALF_WINDOW - 1 : col + HALF_WINDOW, :
            ] = [
                1.0,
                0.0,
                0.0,
            ]

        self.detection_output_image = canvas

    def truncate_candidates(self):
        filtered_faces = list()
        expected_faces = np.argwhere(
            np.array(self.last_stage_info[1]) > self.last_stage_threshold
        )
        for i in range(len(self.last_stage_info[0])):
            if [i] in expected_faces:
                filtered_faces.append(self.last_stage_info[0][i])

        self.render_candidates(self.detection_thumbnail_image, filtered_faces)


"""
Give me:
    - self.detection_original_float -> input
    - self.detection_output_image -> output
"""
