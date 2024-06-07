## To-Do

### User Interface / User Experience:

- Open the same image more than once.
- When you expand or collapse the control panel, the expanded or the collapsed view of the effects menu appears for a second.
- Themes Feature.
- Make it responsive.
- The error message due to invalid types.
- The error message "convert to grayscale" appears even if the image is gray.
- Make the sidebar as a class and use it in the UI file directly.
- Add a toolbar that contains some selection tools and a zoom tool
- Set minimum and maximum width for each section. || the resizing after adding a new tab.
- Update the user guide: Add a tree of the effects. When an item is clicked, it show a gif and basic explanation of how to use it. (Controls Option)
- Add a splitter to the UI so the four sections become resizable

### Backend:

- Threading and a progress bar
- Add an option to reshow the hybrid viewport after closing it.
- Change the threshold of plotting the input and output images vertically.
- Hybrid images: when you check the "image 01" radio button, the "low-pass" and "high-pass" goes unchecked.
- Histogram Tabs
  - Openning multiple histogram tabs for the same image
  - Defining to which image does the tab belong to in its title probably
- Make sure the parameters controllers (e.g. sliders and spinboxes) in each processing technique are set to the default values defined in the effect class.
- When I used SNAKE, it saved the result. In the same run, I opened buildings image to see the Hough line algorithm. It updated the SNAKE gif with the buildings input and output image.
- You can switch to functional programming in case of "Low-Pass Filters" since it is commonly used by other effects.

### General:

- Update the requirements file [if needed]
- Redesign this MessageBox -> import images in hybrid images mode.
- Update ui for the edge detecttion. [I don't remember i made it or not]
- Review the code repetition: the conversion to grayscale in all effects
- Review and update The documentation of the processing techniques classes
- Check Threshold in plotting images vertically or horizontally in the viewport.

## Done:

### UI/UX/Backend:

- Drag and Drop images.
- Standard Images Library.
- Clean the UI file, make it readable
- Reduce the white margins of the canvas.
- First Tab (Main Viewport) is not closable.
- Save the imported or dropped image history.
- Change the plotting orientation depending on the image shape.
- Add title to each subplot, whether it is an input or output or ...
- Add an error message if the user is trying to add an effect to an empty viewport.
- Add history feature so the user is able to open more than one image.
- Add a button to:
  - reset all app.
  - remove all the effects applied so far in the table widget. [Need_a_fix:to_remove_effects_from_the_table] [Fixed]
  - close current image. [won't make, not important I guess. if any thing was important in the first place]
- Saving Function:
  - Current image.
  - All images in the history.
  - "Save as" option for the current image.
- Update all icons to one style. [Skipped]
- Fix: when you open two images, all added effects will get added to the last opened image. [Fixed]
- Make sure the cursor changes with every possible interaction, e.g. buttons, sliders,... etc
- [Corner_Case] when you delete history, if you clicked add noise for example, it will call the last opened image.
- Add a View menu to show or hide different sections of the app (if needed). [Not_Needed]
- Try to make the image ??? Didn't complete the sentence for some reason and i don't remember what the idea was.
- Make a separate file for the helper funcitons.
- Add samples for the Hough Transform in the Tools menu.
- Saving Function: [Cancelled:No-Need]
  - Noise texture for the three types of noise
  - Images after each effect [this option is inside each effect's page].
- Pick specific images to compare them from different effects pipeline && Similarity Test. [Cancelled:No-Need]
- Cumulative/Non-Cumulative mode for the added effects. [Unachievable - Each processing technique has a distinct purpose]
- Added Effects -> Table Functionalities: [Cancelled]
  - Show/Hide
  - Edit
  - Change Order if Cumulative
  - Delete
- Add a button to: [Cancelled]
  - tick or untick the visibility of all added effects in the table widget.
- Fix the orientation of the canvas titles [passed after reducing the white margin around the canvas]
- Save the output image of the edge detector.
- Remove the Cumulative option and the table!

### Digital Image Processing Functions:

- Save the converted grayscale image in the grayscale_image attribute of the class Image.

**Task 01:**

- Additive Noise
  - Uniform
  - Gaussian
  - Salt & Pepper
- Low-Pass Filters
  - Average
  - Gaussian
  - Median
- Edge Detection
  - Sobel
  - Roberts
  - Prewitt
  - Canny
- Draw Histogram and Distribution Curve
- Equalizer Histogram
- Image Equalization
- Image Normalization
- Local and Global Thresholding
- Convert to Grayscale, plot R, G, and B histograms with its distribution function
- Frequency domain filters (high-pass & low-pass)
- Hybrid Images

**Task 02:**

- Active Contour (SNAKE)
- Hough Transform:
  - Line
  - Circle
  - Ellipse

**Task 03:**

- Feature Extraction
  - Harris Operator
  - Lambda-Minus
- SIFT
- Image Matching using
  - Sum of Squared Differences (SSD)
  - Normalized Cross Corrleations

**Task 04:**

- For gray images, and both global & local ways:
  - Optimal Thresholding
  - OTSU Thresholding
  - Spectral Thresholding
- Map RGB to LUV
- For color images:
  - K-means
  - Region Growing
  - Agglomerative Clustering
  - Mean Shift

**Task 05:**

- Detect Faces (color or grayscale)
- Recognize faces based on PCA/Eigen analysis
- Report performance and plot ROC curve
