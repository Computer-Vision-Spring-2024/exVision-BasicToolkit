import numpy as np
from Classes.EffectsWidgets.SnakeGroupBox import SnakeGroupBox
from Classes.ExtendedWidgets.DoubleClickPushButton import QDoubleClickPushButton
from PyQt5.QtCore import pyqtSignal
import matplotlib.pyplot as plt
import imageio

class Snake(QDoubleClickPushButton):
    _instance_counter = 0
    attributes_updated = pyqtSignal(np.ndarray)

    def __init__(self, imageData, ui, parent=None, *args, **kwargs):
        super(Snake, self).__init__(parent)
        # For naming the instances of the effect
        Snake._instance_counter += 1
        self.title = f"Snake.{Snake._instance_counter:03d}"
        self.setText(self.title)  # Set the text of the button to its title
        self.ui= ui
        self.ui.main_viewport_figure_canvas.mpl_connect('motion_notify_event', self.onmove)
        self.ui.main_viewport_figure_canvas.mpl_connect('button_press_event', self.onpress)
        self.ui.main_viewport_figure_canvas.mpl_connect('button_release_event', self.onrelease)
        self.ax1= None
        self.ax2= None
        # Attributes
        # The group box that will contain the effect options
        self.snake_groupbox = SnakeGroupBox(self.title)
        self.snake_groupbox.setVisible(False)
        self.grayscale_image = imageData  # The image that the filter will be applied to
        # Calculate the default filtered image
        self.output_image = self.grayscale_image
        self.display_image()
        # Pass the FreqFilters instance to the FreqFiltersGroupBox class
        self.snake_groupbox.snake_effect = self
        # Store the attributes of the effect to be easily stored in the images instances.
        # self.attributes = self.attributes_dictionary()
        self.contour = np.array([])
        self.drawing = False
        self.window_size = 3
        self.ALPHA = 0.5
        self.BETA = 1
        self.GAMA = 0.2
        self.frames = []
        self.num_iterations = 20 
        self.frames= []
        self.filtered_image= self.output_image
    
    # Methods
    
    def display_image(self):
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
            height, width= self.grayscale_image.shape
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

            self.ax1.imshow(self.grayscale_image, cmap="gray")
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
            plt.ion()
            
    def resample_contour(self, contour, threshold_distance):
        resampled_contour = [contour[0]]  
        current_point = contour[0]
        
        for i in range(1, len(contour)):
            next_point = contour[i]
            distance = np.sqrt(np.sum((next_point - current_point) ** 2))  
            if distance >= threshold_distance:
                num_segments = distance / threshold_distance 
                if num_segments > 1:
                    for j in range(1, int(num_segments)):
                        t = j / num_segments
                        interpolated_point = current_point + t * (next_point - current_point)
                        resampled_contour.append(interpolated_point)
                    distance= np.sqrt(np.sum((next_point - resampled_contour[-1]) ** 2))
                    if distance > 0.6 * 2* threshold_distance :
                        midpoint = (resampled_contour[-1] + next_point) / 2
                        resampled_contour.append(midpoint)
                resampled_contour.append(next_point)
                current_point = next_point  
        self.distances= np.sqrt(np.sum(np.diff(resampled_contour, axis=0)**2, axis=1))
        distance = np.sqrt(np.sum((resampled_contour[0] - resampled_contour[-1]) ** 2))
        if distance< threshold_distance:
            resampled_contour.pop()
        resampled_contour= resampled_contour[::4]
        return np.array(resampled_contour)


    def onmove(self, event):
        if self.drawing and event.inaxes == self.ax1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.contour = np.vstack((self.contour, [x, y])) 
            if len(self.contour) > 1:
                self.ax1.plot([self.contour[-2, 0], x], [self.contour[-2, 1], y], 'r-') 
            plt.draw()

    def onpress(self, event):
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
                self.ax1.plot(resampled_contour[:, 0], resampled_contour[:, 1], 'ro')
                self.contour = np.array(resampled_contour, dtype=int)
                
                
    def compute_internal_energy(self, control_idx, neighbour_pos):
        prev_idx = control_idx - 1 if control_idx > 0 else len(self.contour) - 1
        next_idx = control_idx + 1 if control_idx < len(self.contour) - 1 else 0
        # if the control_pos = neighbour_pos, then i'm compute the internal energy of the control point. (how it vote to the overall energy in the contour) 

        # finite difference way of computation.
        E_elastic = abs(self.contour[next_idx, 0] - neighbour_pos[0]) + abs(self.contour[next_idx, 1] - neighbour_pos[1])
        E_smooth = abs(self.contour[next_idx, 0] - 2 * neighbour_pos[0] + self.contour[prev_idx, 0]) + abs(self.contour[next_idx, 1] - 2 * neighbour_pos[1] + self.contour[prev_idx, 1])

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


    def update_contour(self, image_gradient , window_size ,alpha = 1, beta = 0.5 ,gama = 1):

        for control_idx, control_point in enumerate(self.contour):
            neighbour_grad, neighbour_indices =  self.get_neighbours_with_indices(image_gradient, control_point, window_size)
            external_energy_neighbours = neighbour_grad * gama * -1 
            internal_energy_neighbour = np.zeros_like(neighbour_grad)
            for row in range(neighbour_indices.shape[0]):
                for col in range(neighbour_indices.shape[1]):
                    E_elastic, E_smooth = self.compute_internal_energy(control_idx,neighbour_indices[row,col])
                    internal_energy_neighbour[row, col] = alpha * E_elastic + beta * E_smooth


            overall_energy_neighbours =   external_energy_neighbours + internal_energy_neighbour 
            min_energy = np.argmin(overall_energy_neighbours)

            i, j = np.unravel_index(min_energy, overall_energy_neighbours.shape)

            i_actual, j_actual = neighbour_indices[i,j]

            self.contour[control_idx] = [i_actual, j_actual]
        return  self.contour 

    def diplay_output(self):
        for _ in range(self.num_iterations):
            print("update")
            self.contour = self.update_contour(self.filtered_image, self.window_size, self.ALPHA, self.BETA , self.GAMA)

            # Clear and redraw the plot
            self.ax2.clear()
            self.ax2.imshow(self.grayscale_image, cmap='gray')
            self.ax2.plot(self.contour[:, 0], self.contour[:, 1], 'ro-')

            # # Save the current frame
            self.ui.main_viewport_figure_canvas.canvas.draw()
            frame = np.array(self.ui.main_viewport_figure_canvas.renderer.buffer_rgba())
            self.frames.append(frame)

        imageio.mimsave('snake_animation.gif', self.frames, fps=10)

        self.ax2.clear()
        self.ax2.imshow(self.grayscale_image, cmap='gray')
        self.ax2.plot(self.contour[:, 0], self.contour[:, 1], 'ro')
        self.ui.main_viewport_figure_canvas.draw()



                            
                            
                    
