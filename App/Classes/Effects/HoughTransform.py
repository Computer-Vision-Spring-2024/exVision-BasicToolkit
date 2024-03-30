import typing as tp
from collections import defaultdict

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from Classes.EffectsWidgets.HoughTransformGroupBox import HoughTransformGroupBox


class HoughTransform(QObject):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(
        self,
        type,
        imageData,
        grayscale_image,
        edgedImageData,
        parent=None,
        *args,
        **kwargs,
    ):
        super(HoughTransform, self).__init__(parent)

        # For naming the instances of the effect
        HoughTransform._instance_counter += 1
        self.title = f"HoughTransform.{HoughTransform._instance_counter:03d}"

        # Attributes
        self.type = type  # The type of boundary the user wants to detect "line", "circle", or "ellipse"
        self.original_image = (
            imageData  # The original image we want to superimpose the detection on
        )
        self.grayscale_image = (
            grayscale_image  # The grayscale image of the original image
        )
        self.edged_image = edgedImageData  # The image after canny edge detection
        self.output_image = (
            self.calculate_hough()
        )  # The image after applying the effect

        # The group box that will contain the effect options
        self.hough_groupbox = HoughTransformGroupBox(self.title)
        self.hough_groupbox.setVisible(False)
        # Pass the HoughTransform instance to the HoughTransformGroupBox class
        self.hough_groupbox.hough_transform = self

        # Connect the signal of the group box to the update_parameters method
        self.hough_groupbox.hough_type_combo_box.currentIndexChanged.connect(
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
            "output": self.output_image,
            "groupbox": self.hough_groupbox,
            "final_result": self.update_attributes,
        }

    # Methods
    def update_attributes(self):
        """
        Description:
            - Updates the parameters of the hough transform effect depending on
                the associated effect groupbox.
        """
        self.type = self.hough_groupbox.hough_type_combo_box.currentText()
        self.output_image = self.calculate_hough()
        self.attibutes = self.attributes_dictionary()
        self.attributes_updated.emit(self.output_image)

    def calculate_hough(self):
        if self.type == "Line":
            # Default Line Detection Parameters
            rho = 9
            theta = 0.261
            threshold = 101
            lines = self.hough_line(threshold, -np.pi / 2, np.pi / 2, theta, rho)
            lines_img, _ = self.draw_lines(self.original_image, lines, cv2_setup=False)
            return lines_img

        elif self.type == "Circle":
            min_radius = 10
            max_radius = 80
            accumulator_threshold = 52
            min_dist = 30  # Adjust this value as needed

            return self.hough_circle(
                min_radius, max_radius, accumulator_threshold, min_dist
            )

        elif self.type == "Ellipse":
            return self.hough_ellipse()

    def hough_line(
        self,
        threshold: float,
        min_theta: float,
        max_theta: float,
        theta: float,
        rho: float,
    ) -> np.ndarray:
        """
        Description:
            - Detects the lines in the image using the Hough Transform algorithm.

        Args:
            - threshold: The minimum number of votes a line should have to be considered as a real line.
            - min_theta: The minimum angle of the lines to be detected.
            - max_theta: The maximum angle of the lines to be detected.
            - theta: The step size of the angles to be considered.
            - rho: The step size of the distances to be considered.

        Returns:
            - polar_coordinates: The polar coordinates of the detected lines in the form (rho, theta)
        """
        # Initialize the counter matrix in polar coordinates
        diagonal = np.sqrt(
            self.original_image.shape[0] ** 2 + self.edged_image.shape[1] ** 2
        )

        # Compute the values for the thetas and the rhos
        theta_angles = np.arange(min_theta, max_theta, theta)
        rho_values = np.arange(-diagonal, diagonal, rho)
        # Compute the dimension of the accumulator matrix
        num_thetas = len(theta_angles)
        num_rhos = len(rho_values)
        accumulator = np.zeros([num_rhos, num_thetas])

        # Pre-compute sin and cos
        sins = np.sin(theta_angles)
        coss = np.cos(theta_angles)

        # Consider edges only
        xs, ys = np.where(self.edged_image > 0)

        for x, y in zip(xs, ys):
            for t in range(num_thetas):
                # compute the rhos for the given point for each theta
                current_rho = x * coss[t] + y * sins[t]
                # for each rho, compute the closest rho among the rho_values below it
                # the index corresponding to that rho is the one we will increase
                rho_pos = np.where(current_rho > rho_values)[0][-1]
                # rho_pos = np.argmin(np.abs(current_rho - rho_values))
                accumulator[rho_pos, t] += 1

        # Take the polar coordinates most matched
        final_rho_index, final_theta_index = np.where(accumulator > threshold)
        final_rho = rho_values[final_rho_index]
        final_theta = theta_angles[final_theta_index]

        polar_coordinates = np.vstack([final_rho, final_theta]).T

        return polar_coordinates

    # dp to control accumulator size, min_dist to control accepted circles according to the distance bet. their centers
    def hough_circle(self, min_radius, max_radius, accumulator_threshold, min_dist):
        """
        Description:
            - Detects the circles in the image using the Hough Transform algorithm.

        Args:
            - min_radius: The minimum radius of the circles to be detected.
            - max_radius: The maximum radius of the circles to be detected.
            - accumulator_threshold: The minimum number of votes a circle should have to be considered as a real circle.
            - min_dist: The minimum distance between the centers of the detected circles.

        Returns:
            - circle_img: The image with the detected circles superimposed on it.
        """
        # Define Hough space dimensions based on edge_image size and radius range
        height, width = self.edged_image.shape[:2]

        # Define radius range for iteration
        radii = np.arange(min_radius, max_radius + 1)

        circle_candidates = []

        for radius in radii:
            for theta in np.linspace(0, 2 * np.pi, 100):
                circle_candidates.append(
                    (radius, int(radius * np.cos(theta)), int(radius * np.sin(theta)))
                )

        # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not
        # aready present in the dictionary instead of throwing exception.
        accumulator = defaultdict(int)

        # Find edge pixels in the image to get (x,y) in image space
        edge_pixels = np.argwhere(self.grayscale_image > 0)

        for y in range(height):
            for x in range(width):
                if self.edged_image[y][x] != 0:  # white pixel (edge)
                    for r, r_cos, r_sin in circle_candidates:
                        a = x - r_cos
                        b = y - r_sin
                        accumulator[(a, b, r)] += 1  # vote for current candidate

        # Output image with detected lines drawn
        superimposed_circles_image = self.grayscale_image.copy()
        # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold)
        out_circles = []

        # Sort the accumulator based on the votes for the candidate circles
        for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            x, y, r = candidate_circle
            current_vote = votes
            if current_vote >= accumulator_threshold:
                # Shortlist the circle for final result
                out_circles.append((x, y, r, current_vote))
                print(x, y, r, current_vote)

        filtered_circles = []
        for x, y, r, v in out_circles:
            # Exclude circles that are too close of each other
            # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in filtered_circles)
            # Remove nearby duplicate circles based on min_dist
            if all(
                abs(x - xc) > min_dist
                or abs(y - yc) > min_dist
                or abs(r - rc) > min_dist
                for xc, yc, rc, v in filtered_circles
            ):
                filtered_circles.append((x, y, r, v))
        out_circles = filtered_circles

        # Draw shortlisted circles on the output image
        for x, y, r, v in out_circles:
            superimposed_circles_image = cv2.circle(
                superimposed_circles_image, (x, y), r, (0, 255, 0), 3
            )

        return superimposed_circles_image

    def hough_ellipse(self):
        pass

    # Line Transform Helper Functions
    def draw_lines(
        self,
        img: np.ndarray,
        lines: np.ndarray,
        color: tp.List[int] = [0, 0, 255],
        thickness: int = 1,
        cv2_setup: bool = True,
    ) -> tp.Tuple[np.ndarray]:
        """
        Description:
            - Superimpose the boundary detected lines to the original image image

        Args:
            - img: The original image we want to superimpose the lines on
            - lines: The polar coordinates of the detected lines in the form (rho, theta)
            - color: The color of the lines to be drawn, Default is red
            - thickness: The thickness of the lines to be drawn, Default is 1
            - cv2_setup: A flag to determine if the image is in the cv2 setup or not

        Returns:
            - superimposed_lines_image: The image with the detected lines superimposed on it
        """
        superimposed_lines_image = np.copy(img)
        empty_image = np.zeros(img.shape[:2])

        if len(lines.shape) == 1:
            lines = lines[None, ...]

        # Draw found lines
        for rho, theta in lines:
            x0 = self.polar2cartesian(rho, theta, cv2_setup)
            direction = np.array([x0[1], -x0[0]])
            pt1 = np.round(x0 + 1000 * direction).astype(int)
            pt2 = np.round(x0 - 1000 * direction).astype(int)
            empty_image = cv2.line(
                img=empty_image, pt1=pt1, pt2=pt2, color=255, thickness=thickness
            )

        # Keep lower part of each line until intersection
        mask_lines = empty_image != 0
        min_diff = np.inf
        max_line = 0
        for i in range(mask_lines.shape[0]):
            line = mask_lines[i]
            indices = np.argwhere(line)
            if indices[-1] - indices[0] < min_diff:
                min_diff = indices[-1] - indices[0]
                max_line = i

        mask_boundaries = np.zeros_like(empty_image)
        mask_boundaries[max_line:] = 1
        mask = (mask_lines * mask_boundaries).astype(bool)

        superimposed_lines_image[mask] = np.array(color)

        return superimposed_lines_image, mask

    # Function to perform the conversion between polar and cartesian coordinates
    def polar2cartesian(
        self, radius: np.ndarray, angle: np.ndarray, cv2_setup: bool = True
    ) -> np.ndarray:
        if cv2_setup:
            return radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            return radius * np.array([np.sin(angle), np.cos(angle)])
