import numpy as np
import matplotlib.pyplot as plt
from Classes import *
from scipy.interpolate import interp1d
import imageio


image_3 = Image("images.jpeg") # insert the image path 
image_3.convert_to_grayscale()

processor = ImageProcessor()
magnitude = processor.get_edges(image_3, "sobel_3x3", filter_flag=False) # retruns data not image object 
image_4 = Image(image_data = magnitude)
processor.apply_filter(image_4, "gaussian", filter_size= 20, sigma = 10 ) 
image_4.display()


def resample_contour( contour, threshold_distance):
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
    distance = np.sqrt(np.sum((resampled_contour[0] - resampled_contour[-1]) ** 2))
    if distance< threshold_distance:
        resampled_contour.pop()
    resampled_contour= resampled_contour[::4]
    return np.array(resampled_contour)

def plot_contour(ax, contour):
    ax.clear()
    ax.imshow(image_3.original_img, cmap='gray')
    ax.plot(contour[:, 0], contour[:, 1], 'ro-')
    end_points_x = [contour[0, 0], contour[-1, 0]] 
    end_points_y = [contour[0, 1], contour[-1, 1]] 
    ax.plot(end_points_x,end_points_y, 'ro-')
    ax.axis('off')
    plt.tight_layout()



   

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
            num_points = len(contour) // 2   # Change this number as needed
            resampled_contour = resample_contour(contour, 4)
            plot_contour(ax,resampled_contour)
            contour = np.array(resampled_contour, dtype=int)


image = image_3.original_img
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




def compute_internal_energy(contour, control_idx, neighbour_pos):
    prev_idx = control_idx - 1 if control_idx > 0 else len(contour) - 1
    next_idx = control_idx + 1 if control_idx < len(contour) - 1 else 0
    # if the control_pos = neighbour_pos, then i'm compute the internal energy of the control point. (how it vote to the overall energy in the contour) 

    # finite difference way of computation.
    E_elastic = abs(contour[next_idx, 0] - neighbour_pos[0]) + abs(contour[next_idx, 1] - neighbour_pos[1])
    E_smooth = abs(contour[next_idx, 0] - 2 * neighbour_pos[0] + contour[prev_idx, 0]) + abs(contour[next_idx, 1] - 2 * neighbour_pos[1] + contour[prev_idx, 1])

    internal_energy = (E_elastic, E_smooth)

    return internal_energy

def get_neighbours_with_indices(image_gradient, loc, window_size):
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


def update_contour(image_gradient ,contour, window_size ,alpha = 1, beta = 0.5 ,gama = 1):

    for control_idx, control_point in enumerate(contour):
        neighbour_grad, neighbour_indices =  get_neighbours_with_indices(image_gradient, control_point, window_size)

        external_energy_neighbours = neighbour_grad * gama * -1  

        internal_energy_neighbour = np.zeros_like(neighbour_grad)

        for row in range(neighbour_indices.shape[0]):
            for col in range(neighbour_indices.shape[1]):
                E_elastic, E_smooth = compute_internal_energy(contour,control_idx, neighbour_indices[row,col])
                internal_energy_neighbour[row, col] = alpha * E_elastic + beta * E_smooth


        overall_energy_neighbours = external_energy_neighbours + internal_energy_neighbour 

# ------------------------------------ loose -------------------------------
        # min_energy = np.argmin(overall_energy_neighbours) 

        # i, j = np.unravel_index(min_energy, overall_energy_neighbours.shape)

        # i_actual, j_actual = neighbour_indices[i,j]

        # contour[control_idx] = [i_actual, j_actual]
        
#------------------------------------- restricted ---------------------------
        # high time complexity due to sorting 

        sorted_indices = np.argsort(overall_energy_neighbours, axis=None)

        for min_energy_index in sorted_indices: 

            i, j = np.unravel_index(min_energy_index, overall_energy_neighbours.shape)

            i_actual, j_actual = neighbour_indices[i, j]

            # check if the candidate control point is already existent in coutour 
            if not any(np.all(contour == [i_actual, j_actual], axis=1)): 
                contour[control_idx] = [i_actual, j_actual]
                break 

            # else, keep iterating until getting the lowest energy position. 

    return  contour 

     
    
window_size = 3

ALPHA = 1
BETA = 1
GAMA =  0.5 
num_iterations = 10


frames = []

for _ in range(num_iterations):
    print("update")

    contour = update_contour(image_4.manipulated_img, contour, window_size, alpha=ALPHA, beta=BETA ,gama = GAMA)

    plot_contour(ax, contour)

    # # Save the current frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

imageio.mimsave('snake_animation.gif', frames, fps=10)



input("Press Enter to Generate the Chain Code")


# code_lookup_coordiate_system = {
#     0 : list(range(339,361)) + list(range(0,23,1)), 
#     1 : range(23,68), 
#     2 : range(68,113),  
#     3 : range(113,158), 
#     4 : range(158,203), 
#     5 : range(203, 248), 
#     6 : range(248, 293), 
#     7 : range(293,338)
# }

# tailored specifially 
# we have used this lookup because the origin the in the image space in the upper left corner. 
code_lookup_image_space = {
    0 : list(range(339,361)) + list(range(0,23,1)), 
    7 : range(23,68), 
    7 : range(68,113),  
    5 : range(113,158), 
    4 : range(158,203), 
    3 : range(203, 248), 
    2 : range(248, 293), 
    1 : range(293,338)
}



def compute_chain_code(contour):  
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    
        
    chain_code = list()

    for i in range(len(contour[:-1])):
        dx = contour[i + 1][0] - contour[i][0]
        dy = contour[i + 1][1] - contour[i][1]
        slope = round(np.arctan2(dy,dx) * 180/np.pi) # to degrees 
        if slope < 0: 
            slope += 360
        for key,val in code_lookup_image_space.items():
            if slope in val:
                chain_code.append(key)
                break;

    return chain_code

chain_code= compute_chain_code(contour.copy())
print(chain_code)

input("Press Enter Compute the Area")

# Shoelace formula for computing the area enclosed with a set of points of defined coordinate points
def compute_area(contour):
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

area = compute_area(contour)
print(round(area,3))

# test case (square 2x2)
# contour_arc = np.array([
#     [0,0],
#     [2,0],[2,2],[0,2]])

input("Press Enter Compute the Perimeter")

def compute_perimeter(contour):
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    perimeter = np.sum(distances) + np.linalg.norm(contour[-1] - contour[0]) # adding the Eucliden distance between the first and last points
    return perimeter

perimeter = compute_perimeter(contour)
print(round(perimeter,3))

input("Done!!")