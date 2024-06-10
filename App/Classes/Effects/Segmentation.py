import time
from typing import *

import cv2
import numpy as np
from Classes.EffectsWidgets.SegmentationGroupBox import SegmentationGroupBox
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from Classes.Helpers.HelperFunctions import gaussian_weight
from PyQt5.QtCore import pyqtSignal


class Segmentation(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, *args, **kwargs):
        super(Segmentation, self).__init__(parent)

        # For naming the instances of the effect
        Segmentation._instance_counter += 1
        self.title = f"Segmentation.{Segmentation._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title

        # Attributes

        # The group box that will contain the effect options
        self.segmentation_groupbox = SegmentationGroupBox(self.title)
        self.segmentation_groupbox.setVisible(False)

        # Pass the Segmentation instance to the SegmentationGroupbox class
        self.segmentation_groupbox.segmentation_effect = self

        # Connect the signals of the SegmentationGroupBox

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
            "groupbox": self.segmentation_groupbox,
            "final_result": self.update_attributes,
        }

    # Methods
    def update_attributes(self):
        """
        Description:
            - Updates the parameters of the noise effect depending on
                the associated effect groupbox.
        """
        self.val01 = self.segmentation_groupbox.lower_spinbox.value() / 50
        self.val02 = self.segmentation_groupbox.upper_spinbox.value() / 50
        self.type = self.segmentation_groupbox.noise_type_comb.currentText()
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

    ## ============== Region-Growing Methods ============== ##
    def rg_canvas_clicked(self, event):
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            print(
                f"Clicked pixel at ({x}, {y}) with value {self.rg_input_grayscale[y, x]}"
            )

            # Plot a dot at the clicked location
            ax = self.ui.region_growing_input_figure_canvas.figure.gca()
            ax.scatter(
                x, y, color="red", s=10
            )  # Customize the color and size as needed
            self.ui.region_growing_input_figure_canvas.draw()

            # Store the clicked coordinates as seeds
            if self.rg_seeds is None:
                self.rg_seeds = [(x, y)]
            else:
                self.rg_seeds.append((x, y))

    def update_region_growing_threshold(self):
        self.rg_threshold = self.ui.region_growing_threshold_slider.value()
        self.ui.region_growing_threshold.setText(f"Threshold: {self.rg_threshold}")

    def update_rg_window_size(self):
        self.rg_window_size = self.ui.window_size_spinbox.value()

    def apply_region_growing(self):
        """
        Perform region growing segmentation.

        Parameters:
            image (numpy.ndarray): Input image.
            seeds (list): List of seed points (x, y).
            threshold (float): Threshold for similarity measure.

        Returns:
            numpy.ndarray: Segmented image.
        """
        # Initialize visited mask and segmented image
        start = time.time()
        # 'visited' is initialized to keep track of which pixels have been visited (Mask)
        visited = np.zeros_like(self.rg_input_grayscale, dtype=bool)
        # 'segmented' will store the segmented image where each pixel belonging
        # to a region will be marked with the corresponding color
        segmented = np.zeros_like(self.rg_input)

        # Define 3x3 window for mean calculation
        half_window = self.rg_window_size // 2

        # Loop through seed points
        for seed in self.rg_seeds:
            seed_x, seed_y = seed

            # Check if seed coordinates are within image bounds
            if (
                0 <= seed_x < self.rg_input_grayscale.shape[0]
                and 0 <= seed_y < self.rg_input_grayscale.shape[1]
            ):
                # Process the seed point
                region_mean = self.rg_input_grayscale[seed_x, seed_y]

            # Initialize region queue with seed point
            # It holds the candidate pixels
            queue = [(seed_x, seed_y)]

            # Region growing loop
            # - Breadth-First Search (BFS) is used here to ensure
            # that all similar pixels are added to the region
            while queue:
                # Pop pixel from queue
                x, y = queue.pop(0)

                # Check if pixel is within image bounds and not visited
                if (
                    (0 <= x < self.rg_input_grayscale.shape[0])
                    and (0 <= y < self.rg_input_grayscale.shape[1])
                    and not visited[x, y]
                ):
                    # Mark pixel as visited
                    visited[x, y] = True

                    # Check similarity with region mean
                    if (
                        abs(self.rg_input_grayscale[x, y] - region_mean)
                        <= self.rg_threshold
                    ):
                        # Add pixel to region
                        segmented[x, y] = self.rg_input[x, y]

                        # Update region mean
                        # Incremental update formula for mean:
                        # new_mean = (old_mean * n + new_value) / (n + 1)
                        number_of_region_pixels = np.sum(
                            segmented != 0
                        )  # Number of pixels in the region
                        region_mean = (
                            region_mean * number_of_region_pixels
                            + self.rg_input_grayscale[x, y]
                        ) / (number_of_region_pixels + 1)

                        # Add neighbors to queue
                        for i in range(-half_window, half_window + 1):
                            for j in range(-half_window, half_window + 1):
                                if (
                                    0 <= x + i < self.rg_input_grayscale.shape[0]
                                    and 0 <= y + j < self.rg_input_grayscale.shape[1]
                                ):
                                    queue.append((x + i, y + j))

        self.display_image(
            segmented,
            self.ui.sift_output_figure_canvas,
            "SIFT Output",
            False,
            False,
            "off",
        )
        self.plot_rg_output(segmented)
        end = time.time()
        print(f"time = {end - start}")

    def plot_rg_output(self, segmented_image):
        ## =========== Display the segmented image =========== ##
        # Find contours of segmented region
        contours, _ = cv2.findContours(
            cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours on input image
        output_image = self.rg_input.copy()
        cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

        # Display the output image
        self.display_image(
            output_image,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
        )

    def reset_region_growing(self):
        self.rg_seeds = None
        self.rg_threshold = 20
        self.ui.region_growing_threshold_slider.setValue(self.rg_threshold)
        self.ui.region_growing_threshold.setText(f"Threshold: {self.rg_threshold}")
        self.rg_output = self.rg_input
        self.display_image(
            self.rg_input,
            self.ui.region_growing_input_figure_canvas,
            "Input Image",
            False,
        )
        self.display_image(
            self.rg_output,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
        )

    ## ============== K-Means Methods ============== ##
    def get_new_k_means_parameters(self):
        self.n_clusters = self.ui.n_clusters_spinBox.value()
        self.max_iterations = self.ui.k_means_max_iteratation_spinBox.value()
        self.centroid_optimization = self.ui.centroid_optimization.isChecked()

        self.spatial_segmentation_weight = (
            self.ui.spatial_segmentation_weight_spinbox.value()
        )

        self.spatial_segmentation = self.ui.spatial_segmentation.isChecked()
        self.k_means_LUV = self.ui.k_means_LUV_conversion.isChecked()

    def enable_spatial_segmentation(self):
        if self.ui.spatial_segmentation.isChecked():
            self.spatial_segmentation = True
            self.ui.spatial_segmentation_weight_spinbox.setEnabled(True)
        else:
            self.spatial_segmentation = False
            self.ui.spatial_segmentation_weight_spinbox.setEnabled(False)

    def kmeans_segmentation(
        self,
        image,
        max_iterations,
        centroids_color=None,
        centroids_spatial=None,
    ):
        """
        Perform K-means clustering segmentation on an input image.

        Parameters:
        - centroids_color (numpy.ndarray, optional): Initial centroids in terms of color. Default is None.
        - centroids_spatial (numpy.ndarray, optional): Initial centroids in terms of spatial coordinates. Default is None.

        Returns:
        If include_spatial_seg is False:
        - centroids_color (numpy.ndarray): Final centroids in terms of color.
        - labels (numpy.ndarray): Labels of each pixel indicating which cluster it belongs to.

        If include_spatial_seg is True:
        - centroids_color (numpy.ndarray): Final centroids in terms of color.
        - centroids_spatial (numpy.ndarray): Final centroids in terms of spatial coordinates.
        - labels (numpy.ndarray): Labels of each pixel indicating which cluster it belongs to.
        """
        img = np.array(image, copy=True, dtype=float)

        if self.spatial_segmentation:
            h, w, _ = img.shape
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            xy_coords = np.column_stack(
                (x_coords.flatten(), y_coords.flatten())
            )  # spatial coordinates in the features space

        img_as_features = img.reshape(-1, img.shape[2])  # without spatial info included

        labels = np.zeros(
            (img_as_features.shape[0], 1)
        )  # (image size x 1) this array contains the labels of each pixel (belongs to which centroid)

        distance = np.zeros(
            (img_as_features.shape[0], self.n_clusters), dtype=float
        )  # (distance for each colored pixel over the entire clusters)

        # if the centriods have been not provided
        if centroids_color is None:
            centroids_indices = np.random.choice(
                img_as_features.shape[0], self.n_clusters, replace=False
            )  # initialize the centroids
            centroids_color = img_as_features[centroids_indices]  # in terms of color
            if self.spatial_segmentation:
                centroids_spatial = xy_coords[
                    centroids_indices
                ]  # this to introduce restriction in the spatial space of the image

            # Form initial clustering
            if self.centroid_optimization:
                rows = np.arange(img.shape[0])
                columns = np.arange(img.shape[1])

                sample_size = (
                    len(rows) // 16 if len(rows) > len(columns) else len(columns) // 16
                )
                ii = np.random.choice(rows, size=sample_size, replace=False)
                jj = np.random.choice(columns, size=sample_size, replace=False)
                subimage = img[
                    ii[:, np.newaxis], jj[np.newaxis, :], :
                ]  # subimage for redistribute the centriods

                if self.spatial_segmentation:
                    centroids_color, centroids_spatial, _ = self.kmeans_segmentation(
                        subimage,
                        max_iterations // 2,
                        centroids_color=centroids_color,
                        centroids_spatial=centroids_spatial,
                    )
                else:
                    centroids_color, _ = self.kmeans_segmentation(
                        subimage,
                        max_iterations // 2,
                        centroids_color=centroids_color,
                    )

        for _ in range(max_iterations):
            for centroid_idx in range(centroids_color.shape[0]):
                distance[:, centroid_idx] = np.linalg.norm(
                    img_as_features - centroids_color[centroid_idx], axis=1
                )

                if self.spatial_segmentation:
                    distance[:, centroid_idx] += (
                        np.linalg.norm(
                            xy_coords - centroids_spatial[centroid_idx], axis=1
                        )
                        * self.spatial_segmentation_weight
                    )

            labels = np.argmin(
                distance, axis=1
            )  # assign each point in the feature space a label according to its distance from each centriod based on (spatial and color distance)

            for centroid_idx in range(centroids_color.shape[0]):
                cluster_colors = img_as_features[labels == centroid_idx]
                if len(cluster_colors) > 0:  # Check if cluster is not empty
                    new_centroid_color = np.mean(cluster_colors, axis=0)
                    centroids_color[centroid_idx] = new_centroid_color

                    if self.spatial_segmentation:
                        cluster_spatial = xy_coords[labels == centroid_idx]
                        new_centroid_spatial = np.mean(cluster_spatial, axis=0)
                        centroids_spatial[centroid_idx] = new_centroid_spatial

        if self.spatial_segmentation:
            return centroids_color, centroids_spatial, labels
        else:
            return centroids_color, labels

    def apply_k_means(self):
        self.get_new_k_means_parameters()
        if self.spatial_segmentation:
            if self.k_means_LUV:
                self.display_image(
                    self.k_means_luv_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, _, labels = self.kmeans_segmentation(
                    self.k_means_luv_input, self.max_iterations
                )
            else:
                self.display_image(
                    self.k_means_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, _, labels = self.kmeans_segmentation(
                    self.k_means_input, self.max_iterations
                )

        else:
            if self.k_means_LUV:
                self.display_image(
                    self.k_means_luv_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, labels = self.kmeans_segmentation(
                    self.k_means_luv_input, self.max_iterations
                )
            else:
                self.display_image(
                    self.k_means_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, labels = self.kmeans_segmentation(
                    self.k_means_input, self.max_iterations
                )

        self.k_means_output = centroids_color[labels]

        if self.k_means_LUV:
            self.k_means_output = self.k_means_output.reshape(
                self.k_means_luv_input.shape
            )
        else:
            self.k_means_output = self.k_means_output.reshape(self.k_means_input.shape)

        self.k_means_output = (self.k_means_output - self.k_means_output.min()) / (
            self.k_means_output.max() - self.k_means_output.min()
        )
        self.display_image(
            self.k_means_output,
            self.ui.k_means_output_figure_canvas,
            "K-Means Output",
            False,
        )

    ## ============== Mean-Shift Methods ============== ##
    def get_new_mean_shift_parameters(self):
        self.mean_shift_window_size = self.ui.mean_shift_window_size_spinbox.value()
        self.mean_shift_sigma = self.ui.mean_shift_sigma_spinbox.value()
        self.mean_shift_threshold = self.ui.mean_shift_threshold_spinbox.value()

        self.mean_shift_luv = self.ui.mean_shift_LUV_conversion.isChecked()

    def mean_shift_clusters(
        self, image, window_size, threshold, sigma, max_iterations=100
    ):
        """
        Perform Mean Shift clustering on an image.

        Args:
            image (numpy.ndarray): The input image.
            window_size (float): The size of the window for the mean shift.
            threshold (float): The convergence threshold.
            sigma (float): The standard deviation for the Gaussian weighting.

        Returns:
            list: A list of dictionaries representing the clusters. Each dictionary contains:
                - 'points': A boolean array indicating the points belonging to the cluster.
                - 'center': The centroid of the cluster.
        """
        image = (
            (image - image.min()) * (1 / (image.max() - image.min())) * 255
        ).astype(np.uint8)
        img = np.array(image, copy=True, dtype=float)

        img_as_features = img.reshape(
            -1, img.shape[2]
        )  # feature space (each channel elongated)

        num_points = len(img_as_features)
        visited = np.full(num_points, False, dtype=bool)
        clusters = []
        iteration_number = 0
        while (
            np.sum(visited) < num_points and iteration_number < max_iterations
        ):  # check if all points have been visited, thus, assigned a cluster.
            initial_mean_idx = np.random.choice(
                np.arange(num_points)[np.logical_not(visited)]
            )
            initial_mean = img_as_features[initial_mean_idx]

            while True:
                distances = np.linalg.norm(
                    initial_mean - img_as_features, axis=1
                )  # distances

                weights = gaussian_weight(
                    distances, sigma
                )  # weights for computing new mean

                within_window = np.where(distances <= window_size / 2)[0]
                within_window_bool = np.full(num_points, False, dtype=bool)
                within_window_bool[within_window] = True

                within_window_points = img_as_features[within_window]

                new_mean = np.average(
                    within_window_points, axis=0, weights=weights[within_window]
                )

                # Check convergence
                if np.linalg.norm(new_mean - initial_mean) < threshold:
                    merged = False  # Check merge condition
                    for cluster in clusters:
                        if (
                            np.linalg.norm(cluster["center"] - new_mean)
                            < 0.5 * window_size
                        ):
                            # Merge with existing cluster
                            cluster["points"] = (
                                cluster["points"] + within_window_bool
                            )  # bool array that represent the points of each cluster
                            cluster["center"] = 0.5 * (cluster["center"] + new_mean)
                            merged = True
                            break

                    if not merged:
                        # No merge, create new cluster
                        clusters.append(
                            {"points": within_window_bool, "center": new_mean}
                        )

                    visited[within_window] = True
                    break

                initial_mean = new_mean
            iteration_number += 1

        return clusters

    def calculate_mean_shift_clusters(self, image):
        clusters = self.mean_shift_clusters(
            image,
            self.mean_shift_window_size,
            self.mean_shift_threshold,
            self.mean_shift_sigma,
        )
        output = np.zeros(image.shape)

        for cluster in clusters:
            bool_image = cluster["points"].reshape(image.shape[0], image.shape[1])
            output[bool_image, :] = cluster["center"]

        return output

    def apply_mean_shift(self):
        self.get_new_mean_shift_parameters()

        if self.mean_shift_luv:
            self.display_image(
                self.mean_shift_luv_input,
                self.ui.mean_shift_input_figure_canvas,
                "Input Image",
                False,
            )
            self.mean_shift_output = self.calculate_mean_shift_clusters(
                self.mean_shift_luv_input
            )
        else:
            self.display_image(
                self.mean_shift_input,
                self.ui.mean_shift_input_figure_canvas,
                "Input Image",
                False,
            )
            self.mean_shift_output = self.calculate_mean_shift_clusters(
                self.mean_shift_input
            )

        self.mean_shift_output = (
            self.mean_shift_output - self.mean_shift_output.min()
        ) / (self.mean_shift_output.max() - self.mean_shift_output.min())
        self.display_image(
            self.mean_shift_output,
            self.ui.mean_shift_output_figure_canvas,
            "Mean Shift Output",
            False,
        )

        ## ============== Agglomerative Clustering ============== ##
        def get_agglomerative_parameters(self):
            self.downsampling = self.ui.downsampling.isChecked()
            self.agglo_number_of_clusters = (
                self.ui.agglo_num_of_clusters_spinBox.value()
            )
            self.agglo_scale_factor = self.ui.agglo_scale_factor.value()
            self.agglo_initial_num_of_clusters = (
                self.ui.initial_num_of_clusters_spinBox.value()
            )
            self.ui.initial_num_of_clusters_label.setText(
                "Initial Number of Clusters: " + str(self.agglo_initial_num_of_clusters)
            )
            self.distance_calculation_method = (
                self.ui.distance_calculation_method_combobox.currentText()
            )

        def downsample_image(self):
            """
            Description:
                -   Downsample the input image using nearest neighbor interpolation.
            """
            # Get the dimensions of the original image
            height, width, channels = self.agglo_input_image.shape

            # Calculate new dimensions after downsampling
            new_width = int(width / self.agglo_scale_factor)
            new_height = int(height / self.agglo_scale_factor)

            # Create an empty array for the downsampled image
            downsampled_image = np.zeros(
                (new_height, new_width, channels), dtype=np.uint8
            )

            # Iterate through the original image and select pixels based on the scale factor
            for y in range(0, new_height):
                for x in range(0, new_width):
                    downsampled_image[y, x] = self.agglo_input_image[
                        y * self.agglo_scale_factor, x * self.agglo_scale_factor
                    ]

            return downsampled_image

        def agglo_reshape_image(self, image):
            """
            Description:
                -   It creates an array with each row corresponds to a pixel
                    and each column corresponds to a color channel (R, G, B)
            """
            pixels = image.reshape((-1, 3))
            return pixels

        def euclidean_distance(self, point1, point2):
            """
            Description:
                -   Computes euclidean distance of point1 and point2.
                    Noting that "point1" and "point2" are lists.
            """
            return np.linalg.norm(np.array(point1) - np.array(point2))

        def max_clusters_distance_between_points(self, cluster1, cluster2):
            """
            Description:
                -   Computes distance between two clusters.
                    cluster1 and cluster2 are lists of lists of points
            """
            return max(
                [
                    self.euclidean_distance(point1, point2)
                    for point1 in cluster1
                    for point2 in cluster2
                ]
            )

        def clusters_distance_between_centroids(self, cluster1, cluster2):
            """
            Description:
                -   Computes distance between two centroids of the two clusters
                    cluster1 and cluster2 are lists of lists of points
            """
            cluster1_center = np.average(cluster1, axis=0)
            cluster2_center = np.average(cluster2, axis=0)
            return self.euclidean_distance(cluster1_center, cluster2_center)

        def partition_pixel_into_clusters(self, points, initial_k=25):
            """
            Description:
                -   It partitions pixels into self.initial_k groups based on color similarity
            """
            # Initialize a dictionary to hold the clusters each represented by:
            # The key: the centroid color.
            # The value: the list of pixels that belong to that cluster.
            initial_clusters = {}
            # Defining the partitioning step
            # 256 is the maximum value for a color channel
            d = int(256 / (initial_k))
            # Iterate over the range of initial clusters and assign the centroid colors for each cluster.
            # The centroid colors are determined by the multiples of the step size (d) ranging from 0 to 255.
            # Each centroid color is represented as an RGB tuple (j, j, j) where j is a multiple of d,
            # ensuring even distribution across the color space.
            for i in range(initial_k):
                j = i * d
                initial_clusters[(j, j, j)] = []
            # It calculates the Euclidean distance between the current pixel p and each centroid color (c)
            # It then assigns the pixel p to the cluster with the closest centroid color.
            # grops.keys() returns the list of centroid colors.
            # The min function with a custom key function (lambda c: euclidean_distance(p, c)) finds the centroid color with the minimum distance to the pixel p,
            # and the pixel p is appended to the corresponding cluster in the groups dictionary.
            for i, p in enumerate(points):
                if i % 100000 == 0:
                    print("processing pixel:", i)
                nearest_group_key = min(
                    initial_clusters.keys(), key=lambda c: self.euclidean_distance(p, c)
                )
                initial_clusters[nearest_group_key].append(p)
            # The function then returns a list of pixel groups (clusters) where each group contains
            # the pixels belonging to that cluster.
            # It filters out any empty clusters by checking the length of each cluster list.
            return [g for g in initial_clusters.values() if len(g) > 0]

        def fit_clusters(self, points):
            # initially, assign each point to a distinct cluster
            print("Computing initial clusters ...")
            self.clusters_list = self.partition_pixel_into_clusters(
                points, initial_k=self.agglo_initial_num_of_clusters
            )
            print("number of initial clusters:", len(self.clusters_list))
            print("merging clusters ...")

            while len(self.clusters_list) > self.agglo_number_of_clusters:
                # Find the closest (most similar) pair of clusters
                if self.distance_calculation_method == "distance between centroids":
                    cluster1, cluster2 = min(
                        [
                            (c1, c2)
                            for i, c1 in enumerate(self.clusters_list)
                            for c2 in self.clusters_list[:i]
                        ],
                        key=lambda c: self.clusters_distance_between_centroids(
                            c[0], c[1]
                        ),
                    )
                else:
                    cluster1, cluster2 = min(
                        [
                            (c1, c2)
                            for i, c1 in enumerate(self.clusters_list)
                            for c2 in self.clusters_list[:i]
                        ],
                        key=lambda c: self.max_clusters_distance_between_points(
                            c[0], c[1]
                        ),
                    )

                # Remove the two clusters from the clusters list
                self.clusters_list = [
                    c for c in self.clusters_list if c != cluster1 and c != cluster2
                ]

                # Merge the two clusters
                merged_cluster = cluster1 + cluster2

                # Add the merged cluster to the clusters list
                self.clusters_list.append(merged_cluster)

                print("number of clusters:", len(self.clusters_list))

            print("assigning cluster num to each point ...")
            self.cluster = {}
            for cluster_number, cluster in enumerate(self.clusters_list):
                for point in cluster:
                    self.cluster[tuple(point)] = cluster_number

            print("Computing cluster centers ...")
            self.centers = {}
            for cluster_number, cluster in enumerate(self.clusters_list):
                self.centers[cluster_number] = np.average(cluster, axis=0)

        def get_cluster_number(self, point):
            """
            Find cluster number of point
            """
            # assuming point belongs to clusters that were computed by fit functions
            return self.cluster[tuple(point)]

        def get_cluster_center(self, point):
            """
            Find center of the cluster that point belongs to
            """
            point_cluster_num = self.get_cluster_number(point)
            center = self.centers[point_cluster_num]
            return center

        def apply_agglomerative_clustering(self):
            start = time.time()
            if self.downsampling:
                agglo_downsampled_image = self.downsample_image()
            else:
                agglo_downsampled_image = self.agglo_input_image
            self.get_agglomerative_parameters()
            pixels = self.agglo_reshape_image(agglo_downsampled_image)
            self.fit_clusters(pixels)

            self.agglo_output_image = [
                [self.get_cluster_center(pixel) for pixel in row]
                for row in agglo_downsampled_image
            ]
            self.agglo_output_image = np.array(self.agglo_output_image, np.uint8)

            self.display_image(
                self.agglo_output_image,
                self.ui.agglomerative_output_figure_canvas,
                f"Segmented image with k={self.agglo_number_of_clusters}",
                False,
            )

            end = time.time()
            elapsed_time_seconds = end - start
            minutes = int(elapsed_time_seconds // 60)
            seconds = int(elapsed_time_seconds % 60)
            self.ui.agglo_elapsed_time.setText(
                "Elapsed Time is {:02d} minutes and {:02d} seconds".format(
                    minutes, seconds
                )
            )
