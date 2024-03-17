import numpy as np
import matplotlib.pyplot as plt
from Classes import *


# image_3 = Image("old-metal-coins-19664925.jpg")
# image_3.display()
# image_3.convert_to_grayscale()
# processor = ImageProcessor()
# canny_output = processor.get_edges(image_3, "canny") # retruns data not image object 

# image_4 = Image(image_data = canny_output)
# processor.apply_filter(image_4, "gaussian", filter_size=5) # in place operation


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
    global drawing
    if event.button == 1:
        drawing = False

image = plt.imread("output.png")
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


