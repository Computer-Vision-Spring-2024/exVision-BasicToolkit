## Description:
The app UI meant to be similar to Photoshop UI. It has different sections, each with specific purpose and they affect each other in a specific way. This is the challenge.

## To-Do

### User Experience:
- Cumulative/Non-Cumulative mode for the added effects.
- Saving Function:
    - Noise texture for the three types of noise.
    - Images after each effect [this option is inside each effect's page].
- Added Effects -> Table Functionalities:
    - Show/Hide
    - Edit
    - Change Order if Cumulative
    - Delete
- Pick specific images to compare them from different effects pipeline && Similarity Test.
- Open the same image more than once.
- When you expand or collapse the control panel, the expanded or the collapsed view of the effects menu appears for a second.
- Themes Feature.

### Digital Image Processing Functions (DIPFs):
_That means "Is this effect was integrated in the app or not?"._
- Save the converted grayscale image in the grayscale_image attribute of the class Image.
- Hough Transform from scratch:
    - Line
    - Circle
    - Ellipse


### General:
- Report of all the work.
- A requirements file
- Check Threading.
- Add a button to:
    - tick or untick the visibility of all added effects in the table widget.
- Fix the orientation of the canvas titles.
- Check Threshold in plotting images vertically or horizontally in the viewport.
- Make the image library opens another app window not the ordinary file browser.
- Solve the code repetition: the conversion to grayscale in all effects
- Save the output image of the edge detector.
- Update the documentation of the "normalizer" and "equalizer" classes
- Set minimum and maximum width for each section. || the resizing after adding a new tab.
- MessageBox -> import images in hybrid images mode.
- Closing the hybrid viewport -> reshow.
- You need to update the ui for the edge detecttion.
- Hybrid images: when you check the "image 01" radio button, the "low-pass" and "high-pass" goes unchecked.
- Histogram Tabs
    - Openning multiple histogram tabs for the same image 
    - Defining to which image does the tab belong to in its title probably

## Done:

### UX:
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


### Digital Image Processing Functions (DIPFs):
- Add "Additive Noise":
    - Uniform
    - Gaussian
    - Salt & Pepper
- Filter the noisy image using low-pass filters:
    - Average
    - Median
    - Gaussian
- Detect Edges:
    - Sobel
    - Roberts
    - Prewitt & canny
- Equalize the image.
- Draw "Histograms" and "Distribution Curve".
- Equalizer Histograms
- Transformation from color image to grayscale and plot of the RGB Histograms with its distribution function (Cumulative curve that you use it for mapping and histogram equalization).
- Normalize the image.
- Local & Global Thresholding.
- Frequency domain filters (high-pass & low-pass).
- Hybrid images.
- Active Contour (Snake)