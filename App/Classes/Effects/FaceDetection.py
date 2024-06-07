import os

import numpy as np
from Helpers.HelperFunctions import (
    WINDOW_SIZE,
    Location,
    Size,
    integrate_image,
    normalize,
    possible_feature_shapes,
    possible_locations,
    strong_classifier,
    to_float_array,
)

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import pickle
from typing import *

import numpy as np
from Helpers.Features import *
from PIL import Image


## ============== Detection ============== ##
def get_face_detection_parameters(self):
    self.last_stage_threshold = self.ui.last_stage_threshold_spinbox.value()


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
        canvas[row - HALF_WINDOW - 1 : row + HALF_WINDOW, col - HALF_WINDOW - 1, :] = [
            1.0,
            0.0,
            0.0,
        ]
        canvas[row - HALF_WINDOW - 1 : row + HALF_WINDOW, col + HALF_WINDOW - 1, :] = [
            1.0,
            0.0,
            0.0,
        ]
        canvas[row - HALF_WINDOW - 1, col - HALF_WINDOW - 1 : col + HALF_WINDOW, :] = [
            1.0,
            0.0,
            0.0,
        ]
        canvas[row + HALF_WINDOW - 1, col - HALF_WINDOW - 1 : col + HALF_WINDOW, :] = [
            1.0,
            0.0,
            0.0,
        ]

    self.detection_output_image = canvas
    self.display_image(
        self.detection_output_image,
        self.ui.detection_output_figure_canvas,
        "Output Image",
        False,
    )


def truncate_candidates(self):
    filtered_faces = list()
    expected_faces = np.argwhere(
        np.array(self.last_stage_info[1]) > self.last_stage_threshold
    )
    for i in range(len(self.last_stage_info[0])):
        if [i] in expected_faces:
            filtered_faces.append(self.last_stage_info[0][i])

    self.render_candidates(self.detection_thumbnail_image, filtered_faces)
