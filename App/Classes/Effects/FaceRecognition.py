import os

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from typing import *

import cv2
from Classes.Effects.PCA_class import PCA_class
from Classes.EffectsWidgets.FaceRecognitionGroupBox import FaceRecognitionGroupBox
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from Classes.Helpers.Features import *
from PyQt5.QtCore import pyqtSignal


class FaceRecognition(QDoubleClickPushButton):
    _instance_counter = 0

    def __init__(self, parent=None, *args, **kwargs):
        super(FaceRecognition, self).__init__(parent)

        # For naming the instances of the effect
        FaceRecognition._instance_counter += 1
        self.title = f"FaceRecognition.{FaceRecognition._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes
        self.PCA_test_image_index = 30
        self.PCA_weights = None
        self.PCA_eigen_faces = None
        self.face_recognition_threshold = 2900
        # Configured by the user
        self.structure_number = "one"  # Dataset folder, containing subfolders named after subjects, each containing a minimum of 5 images, with extra images limited to the quantity of the smallest subject folder.
        self.dataset_dir = "App/Resources/FaceRecognition"
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

        # The group box that will contain the effect options
        self.face_recognition_groupbox = FaceRecognitionGroupBox(self.title)
        self.face_recognition_groupbox.setVisible(False)

        # Pass the FaceRecognition instance to the FaceRecognitionGroupBox class
        self.face_recognition_groupbox.face_recognition_effect = self

        # Connect the signals of the FaceRecognition groupbox
        self.face_recognition_groupbox.toggle_query.clicked.connect(
            self.toggle_PCA_test_image
        )
        self.face_recognition_groupbox.apply_face_recognition.clicked.connect(
            self.apply_PCA
        )

    # Setters

    # Methods
    def update_attributes(self):
        """
        Description:
            -   Updates the parameters of the noise effect depending on
                the associated effect groupbox.
        """
        self.val01 = self.face_recognition_groupbox.lower_spinbox.value() / 50
        self.val02 = self.face_recognition_groupbox.upper_spinbox.value() / 50
        self.type = self.face_recognition_groupbox.noise_type_comb.currentText()
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

    ## ============== PCA ============== ##
    def store_dataset_method_one(self, dataset_dir):
        self.faces_train = dict()
        self.faces_test = dict()

        # Initialize a variable to store the size of the first image
        self.first_image_size = None

        for subject in os.listdir(dataset_dir):
            images = []
            if subject == "no match":
                # Add to self.faces_test['no match']
                subject_dir = os.path.join(dataset_dir, subject)
                self.faces_test[subject] = [
                    cv2.imread(
                        os.path.join(subject_dir, filename), cv2.IMREAD_GRAYSCALE
                    )
                    for filename in sorted(os.listdir(subject_dir))
                ]
                continue

            # if subjcet is not 'no match'
            subject_dir = os.path.join(dataset_dir, subject)

            for filename in sorted(os.listdir(subject_dir)):
                image = cv2.imread(
                    os.path.join(subject_dir, filename), cv2.IMREAD_GRAYSCALE
                )

                # If first_image_size is None, this is the first image
                # So, store its size and don't resize it
                if self.first_image_size is None:
                    self.first_image_size = image.shape

                images.append(image)

            # Warning for the user that the minimum number of faces per subject is 5
            if len(images) >= 5:
                # Split the data: 80% for training, 20% for testing
                split_index = int(len(images) * 0.8)
                self.faces_train[subject] = images[:split_index]
                self.faces_test[subject] = images[split_index:]

        # Resize images of 'no match' to match the size of the first image
        if "no match" in self.faces_test:
            for i, image in enumerate(self.faces_test["no match"]):
                self.faces_test["no match"][i] = cv2.resize(
                    image, (self.first_image_size[1], self.first_image_size[0])
                )

        return self.faces_train, self.faces_test

    def store_dataset(
        self, structure_number: str = "one", dataset_dir: str = None
    ) -> None:

        # Define a dictionary that maps structure numbers to functions
        methods = {
            "one": self.store_dataset_method_one,
        }

        # Get the function from the dictionary
        func = methods.get(structure_number)

        # Check if the function exists
        if func is not None:
            # Call the function
            func(dataset_dir)
        else:
            print(f"No function named 'store_dataset_method_{structure_number}' found.")

    def train_pca(self, faces_train: dict):

        faces_train_pca = faces_train.copy()
        # Use list comprehension to flatten images and create labels
        train_faces_matrix = []
        self.train_faces_labels = []
        for subject, images in faces_train_pca.items():
            train_faces_matrix.extend(img.flatten() for img in images)
            self.train_faces_labels.extend([subject] * len(images))
        self.train_faces_matrix = np.array(train_faces_matrix)
        # Create instance of the class
        pca = PCA_class().fit(self.train_faces_matrix, "svd")
        sorted_eigen_values = np.sort(pca.explained_variance_ratio_)[::-1]
        cumulative_variance = np.cumsum(sorted_eigen_values)
        # let's assume that we will consider just 90 % of variance in the data, so will consider just first 101 principal components
        upto_index = np.where(cumulative_variance < 0.9)[0][-1]  # the last one
        no_principal_components = upto_index + 1
        self.PCA_eigen_faces = pca.components[:no_principal_components]
        self.PCA_weights = (
            self.PCA_eigen_faces
            @ (
                self.train_faces_matrix - np.mean(self.train_faces_matrix, axis=0)
            ).transpose()
        )
        return self

    def recognise_face(self, test_face: np.ndarray):

        test_face_to_recognise = test_face.copy()
        if test_face_to_recognise.shape != self.first_image_size:
            test_face_to_recognise = cv2.resize(
                test_face_to_recognise, self.first_image_size
            )
        test_face_to_recognise = test_face_to_recognise.reshape(1, -1)
        test_face_weights = (
            self.PCA_eigen_faces
            @ (
                test_face_to_recognise - np.mean(self.train_faces_matrix, axis=0)
            ).transpose()
        )
        distances = np.linalg.norm(
            self.PCA_weights - test_face_weights, axis=0
        )  # compare row wise
        best_match = np.argmin(distances)
        best_match_subject, best_match_subject_distance = (
            self.train_faces_labels[best_match],
            distances[best_match],
        )
        if best_match_subject_distance > self.face_recognition_threshold:
            best_match_subject = "no match"
        else:
            best_match_subject = best_match_subject

        return best_match_subject, best_match_subject_distance, best_match

    def test_faces_and_labels(self, test_faces_dict: dict) -> (list, list):  # type: ignore
        test_faces_dictionary = test_faces_dict.copy()
        # Flatten the test faces and create corresponding labels
        test_faces = []
        test_labels = []
        for subject, faces in test_faces_dictionary.items():
            for face in faces:
                test_faces.append(face)
                # Encode the labels, no match -> 0, otherwise -> 1
                label = 0 if subject == "no match" else 1
                test_labels.append(label)
        return test_faces, test_labels

    def ROC_curve(self):
        self.ui.ROC_figure.clear()
        test_faces_list = self.test_faces_list.copy()
        test_labels_list = self.test_labels_list.copy()
        # Calculate the distances for each test face
        _, distances, _ = zip(*[self.recognise_face(face) for face in test_faces_list])

        # Define the thresholds based on the distances
        thresholds = np.linspace(min(distances), max(distances), 100)

        tpr_values = []
        fpr_values = []

        for threshold in thresholds:
            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0

            for i, distance in enumerate(distances):
                if distance < threshold and test_labels_list[i] == 1:
                    true_positive += 1
                elif distance < threshold and test_labels_list[i] == 0:
                    false_positive += 1
                elif distance >= threshold and test_labels_list[i] == 1:
                    false_negative += 1
                elif distance >= threshold and test_labels_list[i] == 0:
                    true_negative += 1

            tpr = true_positive / (true_positive + false_negative)
            fpr = false_positive / (false_positive + true_negative)

            tpr_values.append(tpr)
            fpr_values.append(fpr)

        ax = self.ui.ROC_figure.add_subplot(111)

        # Plot onto the subplot
        ax.plot(fpr_values, tpr_values)
        ax.plot([0, 0.93], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")

        # Highlight the 2900 threshold
        threshold_2900_tpr = tpr_values[np.argmin(np.abs(thresholds - 2900))]
        threshold_2900_fpr = fpr_values[np.argmin(np.abs(thresholds - 2900))]
        ax.scatter(threshold_2900_fpr, threshold_2900_tpr, color="red")
        ax.text(threshold_2900_fpr, threshold_2900_tpr, "Threshold 2900")

        self.ui.ROC_figure_canvas.draw()

    def apply_PCA(self):
        self.ui.PCA_output_figure.clear()
        test_image = self.PCA_test_img.copy()

        best_match_subject, best_match_subject_distance, best_match_indx = (
            self.recognise_face(test_image)
        )
        if best_match_subject_distance < self.face_recognition_threshold:
            # Visualize
            self.display_image(
                self.train_faces_matrix[best_match_indx].reshape(self.first_image_size),
                self.ui.PCA_output_figure_canvas,
                f"Best match:{best_match_subject}",
                True,
            )
        else:
            self.display_image(
                np.full_like(
                    self.train_faces_matrix[0].reshape(self.first_image_size),
                    255,
                    dtype=np.uint8,
                ),
                self.ui.PCA_output_figure_canvas,
                "No matching subject",
                True,
            )
        self.ui.PCA_output_figure_canvas.draw()

    def toggle_PCA_test_image(self):
        self.ui.PCA_output_figure.clear()
        self.PCA_test_image_index += 1
        test_labels_list = self.test_labels_list.copy()
        self.PCA_test_image_index = self.PCA_test_image_index % len(test_labels_list)
        self.PCA_test_img = self.test_faces_list[self.PCA_test_image_index]
        self.display_image(
            self.PCA_test_img,
            self.ui.PCA_input_figure_canvas,
            "Query",
            True,
        )
