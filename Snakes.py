import numpy as np
import matplotlib.pyplot as plt
from Classes import *
from scipy.interpolate import interp1d
import imageio


image_3 = Image("4-B4W5KT.jpg")
image_3.display()
image_3.convert_to_grayscale()
processor = ImageProcessor()
magnitude, gradient = processor.get_edges(image_3, "sobel_3x3", filter_flag=False) # retruns data not image object 

# image_4 = Image(image_data = canny_output)
# image_4.display()


def resample_contour(contour, num_points):
    # Calculate the cumulative distance along the contour
    cumulative_distance = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    
    # Normalize the cumulative distance to range [0, 1]
    normalized_distance = cumulative_distance / cumulative_distance[-1]
    
    # Ensure that normalized_distance has the same length as contour
    normalized_distance = np.linspace(0, 1, len(contour))
    
    # Create an interpolation function for each coordinate
    interp_func_x = interp1d(normalized_distance, contour[:, 0], kind='cubic')
    interp_func_y = interp1d(normalized_distance, contour[:, 1], kind='cubic')
    
    # Generate evenly spaced samples along the contour
    t = np.linspace(0, 1, num_points)
    resampled_contour = np.column_stack((interp_func_x(t), interp_func_y(t)))
    
    return resampled_contour

plt.ion()
def onmove(event):
    global drawing, contour # reference the global variables
    if drawing and event.inaxes == ax:
        x, y = int(round(event.xdata)), int(round(event.ydata))
        contour = np.vstack((contour, [x, y])) 
        if len(contour) > 1:
            ax.plot([contour[-2, 0], x], [contour[-2, 1], y], 'r-') 
        plt.draw()

def onpress(event):
    global drawing, contour # declare them as global variables
    if event.button == 1 and event.inaxes == ax:
        drawing = True
        x, y = event.xdata, event.ydata
        x, y = int(round(x)), int(round(y))  
        contour = np.array([[x, y]])

def onrelease(event):
    global drawing, contour
    if event.button == 1:
        drawing = False
        if len(contour) > 0:
            # Resample the collected contour
            num_points = len(contour)  # Change this number as needed
            resampled_contour = resample_contour(contour, num_points)

            ax.plot(resampled_contour[:, 0], resampled_contour[:, 1], 'ro')
            
            contour = np.array(resampled_contour, dtype=int)

# image = plt.imread("output.png")
# image = image_4.manipulated_img
            
image = magnitude
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.axis('off')
plt.tight_layout()

contour = np.array([])
drawing = False

# Connect event handlers
cid1 = fig.canvas.mpl_connect('motion_notify_event', onmove)
cid2 = fig.canvas.mpl_connect('button_press_event', onpress)
cid3 = fig.canvas.mpl_connect('button_release_event', onrelease)


input("Press Enter when done...")
print(len(contour))

ALPHA = 0.1  # Weight for external energy
BETA = 0.1   # Weight for internal energy
GAMA = 1

def compute_external_energy(gradient, contour):
    external_energy = -1 * gradient[contour[:, 1], contour[:, 0]]
    return external_energy

def compute_internal_energy(contour, alpha= 0.1, beta = 0.1):
    E_elastic = np.zeros(len(contour))
    E_smooth = np.zeros(len(contour))
    for i in range(len(contour)):
        prev_idx = i - 1 if i > 0 else len(contour) - 1
        next_idx = i + 1 if i < len(contour) - 1 else 0
        E_elastic[i] = abs(contour[next_idx, 0] - contour[i, 0]) + abs(contour[next_idx, 1] - contour[i, 1])
        E_smooth[i] = abs(contour[next_idx, 0] - 2 * contour[i, 0] + contour[prev_idx, 0]) + abs(contour[next_idx, 1] - 2 * contour[i, 1] + contour[prev_idx, 1])
    
    internal_energy = alpha * E_elastic + beta * E_smooth
    return internal_energy

def update_contour(contour, external_energy, internal_energy, gama = 0.1):
    total_energy = gama * external_energy +  internal_energy
    new_contour = contour.copy()
    for i in range(len(contour)):
        min_energy_idx = np.argmin(total_energy)
        new_contour[i] = contour[min_energy_idx]
        total_energy[min_energy_idx] = np.inf  # Mark as visited
    return new_contour



frames = []
num_iterations = 100 # You can adjust the number of iterations
for _ in range(num_iterations):
    print("update")
    external_energy = compute_external_energy(magnitude, contour)
    
    # Compute internal energy
    internal_energy = compute_internal_energy(contour, alpha=ALPHA, beta=BETA)
    
    # Update contour points
    contour = update_contour(contour, external_energy, internal_energy, gama = GAMA)
    
    # Clear and redraw the plot
    ax.clear()
    ax.imshow(magnitude, cmap='gray')
    ax.plot(contour[:, 0], contour[:, 1], 'ro')
    # # plt.draw()
    # plt.pause(0.01)  # Add a short pause to visualize the changes


    # # Save the current frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

imageio.mimsave('snake_animation.gif', frames, fps=30)


# fig_2, ax_2 = plt.subplots()
# ax_2.imshow(magnitude, cmap='gray')
# ax_2.plot(contour[:, 0], contour[:, 1], 'ro')
# ax.axis('off')
# plt.tight_layout()

input("")