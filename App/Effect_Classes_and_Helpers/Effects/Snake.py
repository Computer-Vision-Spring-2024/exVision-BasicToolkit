import imageio
import matplotlib.pyplot as plt
import numpy as np
from Effect_Classes_and_Helpers.Effects.Filter import Filter
from Effect_Classes_and_Helpers.EffectsWidgets.SnakeGroupBox import SnakeGroupBox
from Effect_Classes_and_Helpers.ExtendedWidgets.DoubleClickPushButton import (
    QDoubleClickPushButton,
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog


class Snake(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(
        self, img_data, grayimageData, grayscale_flag, ui, parent=None, *args, **kwargs
    ):
        super(Snake, self).__init__(parent)
        # For naming the instances of the effect
        Snake._instance_counter += 1
        self.title = f"Snake.{Snake._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title
        self.ui = ui
        self.ui.main_viewport_figure_canvas.mpl_connect(
            "motion_notify_event", self.onmove
        )
        self.ui.main_viewport_figure_canvas.mpl_connect(
            "button_press_event", self.onpress
        )
        self.ui.main_viewport_figure_canvas.mpl_connect(
            "button_release_event", self.onrelease
        )
        self.ax1 = None  # input image subplot
        self.ax2 = None  # output image subplot
        # Attributes
        # The group box that will contain the effect options
        self.snake_groupbox = SnakeGroupBox(self.title)
        self.snake_groupbox.setVisible(False)
        self.snake_groupbox.iterations_spinbox.editingFinished.connect(
            self.update_displayed_result
        )
        self.snake_groupbox.ALPHA_spinbox.editingFinished.connect(
            self.update_displayed_result
        )
        self.snake_groupbox.BETA_spinbox.editingFinished.connect(
            self.update_displayed_result
        )
        self.snake_groupbox.GAMMA_spinbox.editingFinished.connect(
            self.update_displayed_result
        )
        self.grayscale_image = grayimageData  # The grayscale image
        self.original_img = img_data  # The original image
        self.filtered_image = (
            self.get_filtered_image()
        )  # The image after applying sobel and gaussian filter
        # To determine whether the user already converted the original image to grayscale or not, if converted
        # display the output image as grayscale image
        self.grayscale_flag = grayscale_flag
        # Set the output image either as RGB or grayscale
        self.determine_output_image_color_mode()
        self.display_image()
        # Pass the Snake instance to the SnakeGroupBox class
        self.snake_groupbox.snake_effect = self
        self.contour = np.array([])
        self.drawing = False
        self.window_size = 3
        self.ALPHA = 0.5
        self.BETA = 1
        self.GAMA = 0.2
        self.frames = []
        self.num_iterations = 14
        self.output_contour = np.array([])
        self.chain_code = None
        # Store the attributes of the effect to be easily stored in the images instances.
        self.attributes = self.attributes_dictionary()
        # we have used this lookup because the origin in the image space is the upper left corner.
        self.code_lookup_image_space = {
            0: list(range(339, 361)) + list(range(0, 23, 1)),
            7: range(23, 68),
            6: range(68, 113),
            5: range(113, 158),
            4: range(158, 203),
            3: range(203, 248),
            2: range(248, 293),
            1: range(293, 338),
        }
        self.snake_groupbox.export_button.clicked.connect(self.export_chain_code)

    # Setters
    def attributes_dictionary(self):
        """
        Description:
            - Returns a dictionary containing the attributes of the effect.
        """
        return {
            "Alpha": self.ALPHA,
            "Beta": self.BETA,
            "Gamma": self.GAMA,
            "Iterations": self.num_iterations,
            "groupbox": self.snake_groupbox,
            "final_result": self.display_final_result,
        }

    # Methods

    def determine_output_image_color_mode(self):
        """
        Description:
            - Determine if the user wants the output image to be grayscale or RGB.
        """
        if self.grayscale_flag == 0:
            self.output_image = self.original_img
        else:
            self.output_image = self.grayscale_image

    def display_image(self):
        """
        Description:
            - Displays the input and output images in the main canvas figure subplots.
        """
        # Clear the previous plot
        self.ui.main_viewport_figure_canvas.figure.clear()
        # Determine layout based on image dimensions
        height, width = self.grayscale_image.shape
        if (width - height) > 300:  # If width is greater than the height
            self.ax1 = self.ui.main_viewport_figure_canvas.figure.add_subplot(
                211
            )  # Vertical layout
            self.ax2 = self.ui.main_viewport_figure_canvas.figure.add_subplot(212)
        else:  # If height is significantly greater than width
            self.ax1 = self.ui.main_viewport_figure_canvas.figure.add_subplot(
                121
            )  # Horizontal layout
            self.ax2 = self.ui.main_viewport_figure_canvas.figure.add_subplot(122)

        self.ax1.imshow(self.original_img, cmap="gray")
        self.ax1.axis("off")
        self.ax1.set_title("Input Image", color="white")

        self.ax2.imshow(self.output_image, cmap="gray")
        self.ax2.axis("off")
        self.ax2.set_title("Output Image", color="white")

        # Reduce the white margins
        self.ui.main_viewport_figure_canvas.figure.subplots_adjust(
            left=0, right=1, bottom=0.05, top=0.95
        )

        # Redraw the canvas
        self.ui.main_viewport_figure_canvas.draw()

    def resample_contour(self, contour, threshold_distance):
        """
        Description:
        Resample the contour to get appropriate number of equidistance point.

        Args:
            -contour: Array of initial contour points.
            -threshold_distance: Threshold distance for resampling.

        Returns:
            -resampled_contour: Array containing the resampled points.
        """
        # Add the first point of the initial contour to the resampled contour array
        resampled_contour = [contour[0]]
        current_point = contour[0]
        # Loop over the contour points
        for i in range(1, len(contour)):
            next_point = contour[i]
            # Calculate the distance between the current point and the next point and compare it with the threshold
            distance = np.sqrt(np.sum((next_point - current_point) ** 2))
            if distance >= threshold_distance:
                num_segments = distance / threshold_distance
                # if the distance is greater than twice the threshold, perform linear interpolation to get points
                # between those two points at distances equal to the threshold distance
                if num_segments > 1:
                    for j in range(1, int(num_segments)):
                        t = j / num_segments
                        interpolated_point = current_point + t * (
                            next_point - current_point
                        )
                        resampled_contour.append(interpolated_point)
                distance = np.sqrt(np.sum((next_point - resampled_contour[-1]) ** 2))
                # if the distance is 60% close to double the threshold
                if distance > 0.6 * 2 * threshold_distance:
                    # take new point at the middle between the current point and the next point
                    midpoint = (resampled_contour[-1] + next_point) / 2
                    resampled_contour.append(midpoint)
                    resampled_contour.append(next_point)
                else:
                    resampled_contour.append(next_point)
                # update the current point
                current_point = next_point
        # Calculate the distance between the first and last point and compare it to the threshold
        distance = np.sqrt(np.sum((resampled_contour[0] - resampled_contour[-1]) ** 2))
        if distance < threshold_distance:
            # if it is less than the threshold, remove the last point
            resampled_contour.pop()
        # if there is 40 points or more, Drop some points from the final array to reduce its size to be around 20 points
        if len(resampled_contour) >= 40:
            drop_step = int(len(resampled_contour) / 20)
            resampled_contour = resampled_contour[::drop_step]
        return np.array(resampled_contour)

    def onmove(self, event):
        if self.drawing and event.inaxes == self.ax1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.contour = np.vstack((self.contour, [x, y]))
            if len(self.contour) > 1:
                self.ax1.plot([self.contour[-2, 0], x], [self.contour[-2, 1], y], "r-")
            plt.draw()

    def onpress(self, event):
        self.set_up_subplot(self.ax1, self.original_img, "Input Image")
        self.ui.main_viewport_figure_canvas.draw()
        if event.button == 1 and event.inaxes == self.ax1:
            self.drawing = True
            x, y = event.xdata, event.ydata
            x, y = int(round(x)), int(round(y))
            self.contour = np.array([[x, y]])

    def onrelease(self, event):
        if event.button == 1:
            self.drawing = False
            if len(self.contour) > 0:
                # Resample the collected contour
                resampled_contour = self.resample_contour(self.contour, 4)
                # Draw the final equidistance contour points
                self.ax1.plot(resampled_contour[:, 0], resampled_contour[:, 1], "ro")
                # Connect the first point with the last point at which the mouse is released
                end_points_x = [resampled_contour[0, 0], event.xdata]
                end_points_y = [resampled_contour[0, 1], event.ydata]
                self.ax1.plot(end_points_x, end_points_y, "r-")
                self.contour = np.array(resampled_contour, dtype=int)
                self.snake_groupbox.area1_line_edit.setText(
                    self.compute_area(resampled_contour)
                )
                self.snake_groupbox.perimeter1_line_edit.setText(
                    self.compute_perimeter(resampled_contour)
                )
                self.display_output()

    def compute_internal_energy(self, contour, control_idx, neighbour_pos):
        prev_idx = control_idx - 1 if control_idx > 0 else len(contour) - 1
        next_idx = control_idx + 1 if control_idx < len(contour) - 1 else 0
        # if the control_pos = neighbour_pos, then i'm compute the internal energy of the control point. (how it vote to the overall energy in the contour)

        # finite difference way of computation.
        E_elastic = abs(contour[next_idx, 0] - neighbour_pos[0]) + abs(
            contour[next_idx, 1] - neighbour_pos[1]
        )
        E_smooth = abs(
            contour[next_idx, 0] - 2 * neighbour_pos[0] + contour[prev_idx, 0]
        ) + abs(contour[next_idx, 1] - 2 * neighbour_pos[1] + contour[prev_idx, 1])

        internal_energy = (E_elastic, E_smooth)

        return internal_energy

    def get_neighbours_with_indices(self, image_gradient, loc, window_size):
        margin = window_size // 2
        i = loc[0] - margin
        j = loc[1] - margin
        i_start = max(0, i)
        j_start = max(0, j)
        i_end_candidate = i_start + window_size
        i_end = np.min((image_gradient.shape[0], i_end_candidate))

        j_end_candidate = j_start + window_size
        j_end = np.min((image_gradient.shape[1], j_end_candidate))

        neighbour_grad = image_gradient[i_start:i_end, j_start:j_end]

        neighbour_indices = np.zeros_like(neighbour_grad, dtype=tuple)

        for x in range(neighbour_indices.shape[0]):
            for y in range(neighbour_indices.shape[1]):
                neighbour_indices[x, y] = (i_start + x, j_start + y)

        return neighbour_grad, neighbour_indices

    def update_contour(
        self, image_gradient, contour, window_size, alpha=1, beta=0.5, gama=1
    ):

        for control_idx, control_point in enumerate(contour):
            neighbour_grad, neighbour_indices = self.get_neighbours_with_indices(
                image_gradient, control_point, window_size
            )

            external_energy_neighbours = neighbour_grad * gama * -1

            internal_energy_neighbour = np.zeros_like(neighbour_grad)

            for row in range(neighbour_indices.shape[0]):
                for col in range(neighbour_indices.shape[1]):
                    E_elastic, E_smooth = self.compute_internal_energy(
                        contour, control_idx, neighbour_indices[row, col]
                    )
                    internal_energy_neighbour[row, col] = (
                        alpha * E_elastic + beta * E_smooth
                    )

            overall_energy_neighbours = (
                external_energy_neighbours + internal_energy_neighbour
            )

            # ------------------------------------ loose -------------------------------
            # min_energy = np.argmin(overall_energy_neighbours)

            # i, j = np.unravel_index(min_energy, overall_energy_neighbours.shape)

            # i_actual, j_actual = neighbour_indices[i,j]

            # contour[control_idx] = [i_actual, j_actual]

            # ------------------------------------- restricted ---------------------------
            # high time complexity due to sorting

            sorted_indices = np.argsort(overall_energy_neighbours, axis=None)

            for min_energy_index in sorted_indices:

                i, j = np.unravel_index(
                    min_energy_index, overall_energy_neighbours.shape
                )

                i_actual, j_actual = neighbour_indices[i, j]

                # check if the candidate control point is already existent in coutour
                if not any(np.all(contour == [i_actual, j_actual], axis=1)):
                    contour[control_idx] = [i_actual, j_actual]
                    break

                # else, keep iterating until getting the lowest energy position.

        return contour

    def display_output(self):
        self.output_contour = self.contour.copy()
        for _ in range(self.num_iterations):
            self.output_contour = self.update_contour(
                self.filtered_image,
                self.output_contour,
                self.window_size,
                self.ALPHA,
                self.BETA,
                self.GAMA,
            )

            # Clear and redraw the plot
            self.set_up_subplot(self.ax2, self.output_image, "Output Image")
            self.draw_contour(self.ax2, self.output_contour)
            # Save the current frame
            frame = np.array(self.ui.main_viewport_figure_canvas.renderer.buffer_rgba())
            self.frames.append(frame)
        imageio.mimsave("snake_animation.gif", self.frames, fps=10)
        self.set_up_subplot(self.ax2, self.output_image, "Output Image")
        self.draw_contour(self.ax2, self.output_contour)
        self.snake_groupbox.area2_line_edit.setText(
            self.compute_area(self.output_contour)
        )
        self.snake_groupbox.perimeter2_line_edit.setText(
            self.compute_perimeter(self.output_contour)
        )
        self.compute_chain_code()
        self.snake_groupbox.export_label.setText("")

    def get_filtered_image(self):
        """
        Description:
            - Apply sobel and guassian filter to the input image as an initial step in the snake algorithm.
        Returns:
            - The resultant filtered image.

        """
        filter_effect = Filter("Gaussian", "5", 1, self.grayscale_image)
        filtered_image = filter_effect.output_image
        filtered_image = self.get_edges(filtered_image)
        filter_effect.grayscale_image = filtered_image
        filter_effect.sigma = 10
        filter_effect.kernel_size = 20
        filtered_image = filter_effect.calculate_filter()
        return filtered_image

    def compute_gradient_using_convolution(self, image, x_kernel, y_kernel):
        """
        Description:
            - Compute edges and gradient directions of the input image using the specified x and y kernels.

        Parameters:
            - image (numpy.ndarray): The input image.
            - x_kernel (numpy.ndarray): The kernel for computing the x component of the gradient.
            - y_kernel (numpy.ndarray): The kernel for computing the y component of the gradient.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The resultant edges image.
                - The gradient directions image.
        """

        x_component = self.convolve_2d(image, x_kernel)
        y_component = self.convolve_2d(image, y_kernel)
        resultant = np.abs(x_component) + abs(y_component)
        resultant = resultant / np.max(resultant) * 255
        direction = np.arctan2(y_component, x_component)
        return (resultant, direction)

    def convolve_2d(self, image, kernel, mutlipy=True):
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

        if pad_size == 0:
            padded_image = image
            normalize_value = 2
        else:
            # padding the image to include edges
            normalize_value = kernel_size * kernel_size
            padded_image = self.padding_image(
                image, image_width, image_height, pad_size
            )

        output_image = np.zeros_like(image)

        for i in range(image_height):
            for j in range(image_width):
                neighborhood = padded_image[
                    i : i + kernel_size, j : j + kernel_size
                ]  # slice out the region
                # optimization trick
                if mutlipy:
                    output_image[i, j] = np.sum(neighborhood * kernel)
                else:
                    output_image[i, j] = np.sum(neighborhood) * (1 / normalize_value)
        return np.clip(output_image, 0, 255)

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
        padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size))
        padded_image[pad_size:-pad_size, pad_size:-pad_size] = image
        return padded_image

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
        return self.compute_gradient_using_convolution(image, dI_dX, dI_dY)

    def get_edges(self, image):
        """
        Apply the Sobel 3x3 edge detection filter to the input image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            - The resultant edged image.
        """

        edged_image = self.sobel_3x3(image)
        if len(edged_image) == 2:
            return edged_image[0]
        else:
            return edged_image

    def display_final_result(self):
        """
        Description:
            - To easily get back to the final output of the effect after displaying other effects.
        """
        self.display_image()
        if not self.contour.size == 0:
            self.draw_contour(self.ax1, self.contour)
            self.draw_contour(self.ax2, self.output_contour)

    def draw_contour(self, ax, contour):
        """
        Description:
            - To draw the given contour in the given subplot.
        Args:
            - subplot: The subplot that will be updated.
            - contour: The contour that will be drawn.
        """
        ax.plot(contour[:, 0], contour[:, 1], "ro-")
        end_points_x = [contour[0, 0], contour[-1, 0]]
        end_points_y = [contour[0, 1], contour[-1, 1]]
        ax.plot(end_points_x, end_points_y, "ro-")
        self.ui.main_viewport_figure_canvas.draw()

    def set_up_subplot(self, subplot, image, title):
        """
        Description:
            - To clear and redraw any of the two subplots without code repetition.
        Args:
            - subplot: The subplot that will be updated.
            - image: The image that will be displayed.
            - title: The title of the subplot.
        """
        subplot.clear()
        subplot.imshow(image, cmap="gray")
        subplot.set_title(title, color="white")

    def update_displayed_result(self):
        """
        Description:
            - Updates the parameters of the effect and the output depending on
                the associated effect groupbox.
        """
        self.ALPHA = self.snake_groupbox.ALPHA_spinbox.value()
        self.BETA = self.snake_groupbox.BETA_spinbox.value()
        self.GAMA = self.snake_groupbox.GAMMA_spinbox.value()
        self.num_iterations = self.snake_groupbox.iterations_spinbox.value()
        self.frames = []
        self.display_output()

    def compute_chain_code(self):
        if not np.array_equal(self.output_contour[0], self.output_contour[-1]):
            contour = np.vstack((self.output_contour, self.output_contour[0]))
        self.chain_code = list()

        for i in range(len(contour[:-1])):
            dx = contour[i + 1][0] - contour[i][0]
            dy = contour[i + 1][1] - contour[i][1]
            slope = round(np.arctan2(dy, dx) * 180 / np.pi)  # to degrees
            if slope < 0:
                slope += 360
            for key, val in self.code_lookup_image_space.items():
                if slope in val:
                    self.chain_code.append(key)
                    break

    # Shoelace formula for computing the area enclosed with a set of points of defined coordinate points
    def compute_area(self, contour):
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return "{:.1f}".format(area)

    # test case (square 2x2)
    # contour_arc = np.array([
    #     [0,0],
    #     [2,0],[2,2],[0,2]])

    def compute_perimeter(self, contour):
        distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
        perimeter = np.sum(distances) + np.linalg.norm(
            contour[-1] - contour[0]
        )  # adding the Eucliden distance between the first and last points
        return "{:.1f}".format(perimeter)

    def export_chain_code(self):
        if self.chain_code == None:
            self.snake_groupbox.export_label.setText("Chain code is empty!")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chain Code", "", "Text Files (*.txt)"
        )
        if file_path:
            with open(file_path, "w") as f:
                for code in self.chain_code:
                    f.write(str(code) + "\n")
