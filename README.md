# ImageAlchemy
ImageAlchemy - A PyQT5-based image processing application offering a variety of techniques implemented from scratch without external libraries.

![App UI](https://github.com/Computer-Vision-Spring-2024/Task-2/blob/main/README_resources/App_UI.png)


## Table of Contents:
- [Description](#description)
- [Features](#project-features)
- [Quick Preview](#quick-preview)
- [App Flow](#app-flow)
- [Folder Structure](#folder-structure)
- [Executing program](#executing-program)
- [Shortcuts](#shortcuts)
- [Future Updates](#future-updates)
- [Help](#help)
- [Contributors](#contributors)
- [License](#license)

## Description
ImageAlchemy is a comprehensive image processing application developed using PyQT5, providing a range of powerful techniques including grayscale conversion, noise addition, various filters, edge detection, thresholding, equalization, active contouring using Snake Algorithm, hybrid image creation, and Hough transformations. Remarkably, all functionalities are implemented from scratch, avoiding dependency on external libraries like OpenCV or Scikit-Image, ensuring a lightweight and self-contained solution for digital image processing needs.

## Project Features
:white_check_mark: **3 Ways to add an image to the viewport**
- :heavy_check_mark: Using the import button in the **_File_** menu in the menu bar, or use the [shortcut](#shortcuts)
- :heavy_check_mark: Drag and drop an image from your system
- :heavy_check_mark: Pick an image from the library included in the **_Tools_** menu in the menu bar

:white_check_mark: **Image Tree: Full Image History**
- :heavy_check_mark: Open multiple images at the same time and seamlessly browse between them, applying different processing techniques to each.

:white_check_mark: **Full control over the applied effects** (under development)
  <p> Upon applying multiple effects to an image, you can arrange the effects into a chain to get a cumulative pipeline that offers you the following: </p>
  
- :heavy_check_mark: Visualise each effect on its own.
- :heavy_check_mark: Delete a specific effect, whether you are in the mode of accumulation or not. If you are in this mode, the outputs of other effects of the chain will be recalculated.
- :heavy_check_mark: Change the order of the effects to get a new cumulative output.

:white_check_mark: **Digital Image Processing Techniques**
- :heavy_check_mark: Convert to grayscale
- :heavy_check_mark: Add 3 different types of noise: Uniform, Gaussian, and Salt & Pepper
- :heavy_check_mark: Filter Image with 3 different types of filters: Mean, Median, and Gaussian
- :heavy_check_mark: Detect Edges in Images with 6 types of detectors
- :heavy_check_mark: Apply High-pass and Low-pass Frequency Filters to Images
- :heavy_check_mark: Apply Local and Global Threshold to Images
- :heavy_check_mark: Equalize Images
- :heavy_check_mark: Normalize Images
- :heavy_check_mark: Create Hybrid Images
- :heavy_check_mark: Active Contouring, Snake Algorithm
- :heavy_check_mark: Perform Hough Transformations for Lines, Circles, and Ellipses

## Quick Preview

#### Browse Images
![Browse Images](README_resources/Import__gif.gif)
#### Apply and adjust local and global threshold effect
![Apply and adjust local and global threshold effect](README_resources/Thresholding.gif)

## Shortcuts

| Shortcut     | Functionality                                                        |
|--------------|----------------------------------------------------------------------|
| Ctrl+I       | Import Image                                                         |
| Ctrl+S       | Save the output image of the current image as PNG                    |
| Shift+Ctrl+S | Save the output image of the current image and specify the extension |
| Alt+S        | Save all the output images of the opened images                      |
| Ctrl+Q       | Exit the app                                                         |
| Ctrl+C       | Show a guide on how to use the app (Currently unavailable)           |
| Ctrl+H       | Open the documentation of the app (Currently unavailable)            |

## App Flow

![App Flow](https://github.com/Computer-Vision-Spring-2024/Task-2/blob/main/README_resources/App_Flow.png)

## Folder Structure
<table>
  <tr>
    <td><p>

**_Classes_**
- `DialogTheme`: This folder contains only the dialog class that appears to the user when he wants to change the appearance of the app.

- `Effects`: This directory houses classes, each representing a specific image processing effect.

- `EffectsWidgets`: Within this folder are classes inheriting from GroupBox, designed to streamline the integration of effect controllers (e.g., sliders, spinboxes) into the user interface. Upon applying an effect, an instance of the corresponding groupbox is instantiated and added to the UI, allowing users to adjust parameters and observe the resulting changes in real-time.

- `ExtendedWidgets`: This directory contains custom widgets developed to address specific developmental challenges, tailored for internal use by developers rather than end-users.

**_Resources_**
- `Icons`: This folder contains the icons used in the stylesheets of the app.

- `ImagesLibrary`: An assortment of image examples categorized into Classic, Old Classic, Fingerprints, High Resolution, Medical, Special, Sun and Planets, Textures, and Additional. Users can utilize these images to test various effects.

- `HoughSamples`: Sanple images to test the Hough Transformation algorithm. These are organized into 3 subfolders for each type of Hough Transformation: Line, Circle, and Ellipse.

- `Themes`: (under development): Currently in progress, this section will incorporate multiple stylesheets or themes, enabling users to customize the appearance of the application, although this feature is not yet accessible for use.

**_Main Files in the root_**:

`ImageAlchemyUI.py`: The user interface file, housing fixed UI components of the application, which can be dynamically modified during runtime. This serves as the primary file executed to launch the application.

`ImageAlchemyBackend.py (Backend & Image Class)`: This file encapsulates the core functionality of the application, orchestrating interactions between the user interface and backend processing. Upon user selection of an effect, the backend instantiates the corresponding effect class, passes inputs, retrieves output images, and manages image history by creating instances of the image class and storing relevant attributes and applied effects.

`HelperFunctions.py`: As the name suggests, this folder contains the functions that facilitate the application of different effects, especially the Equalizer.</p>
    </td>
    <td>

    --- Folder Structure ---
    [App]
      ├── [Classes]
          ├── [DialogTheme]
              ├── ThemeDialog.py
              └── ThemeDialog.ui
          ├── [Effects]
              ├── EdgeDetector.py
              ├── Equalizer.py
              ├── Filter.py
              ├── FreqFilters.py
              ├── HoughTransform.py
              ├── Hybrid.py
              ├── Noise.py
              ├── Normalize.py
              ├── Snake.py
              └── Thresholding.py
          ├── [EffectsWidgets]
              ├── EdgeDetectorGroupBox.py
              ├── FilterGroupBox.py
              ├── FreqFiltersGroupBox.py
              ├── GrayscaleGroupbox.py
              ├── HistogramGroupbox.py
              ├── HoughTransformGroupBox.py
              ├── HybridGroupBox.py
              ├── NoiseGroupBox.py
              ├── NormalizeGroupBox.py
              ├── SnakeGroupBox.py
              └── ThresholdingGroupBox.py
          └── [ExtendedWidgets]
              ├── CanvasWidget.py
              ├── CustomFrame.py
              ├── CustomTabWidget.py
              ├── DoubleClickPushButton.py
              ├── TableWithMovingRows.py
              ├── TabWidget.py
              └── ThemeDialog.py
      ├── HelperFunctions.py
      ├── ImageAlchemyBackend.py
      ├── ImageAlchemyUI.py
      ├── [Resources]
          ├── [HoughSamples]
              ├── [Circle]
              ├── [Ellipse]
              └── [Line]
          ├── [Icons]
              ├── App_Icon.png
              ├── clear.png
              ├── close.png
              ├── down-arrow.png
              ├── [Effects]
              ├── effects.png
              ├── history.png
              ├── LeftArrows.png
              ├── RightArrows.png
              ├── SliderHandle.png
              ├── undo.png
              ├── up-arrow.png
              └── view.png
          ├── [Other Image Resources]
          └── [Themes]
              ├── BlackTheme.qss
              ├── Stylesheet.qss
              └── WhiteTheme.qss
  README.md
  [README_resources]
      ├── App_Flow.png
      ├── App_UI.png
      ├── Import__gif.gif
      └── Thresholding.gif
  requirements.txt

  </tr>
</table>

## Executing program

To be able to use our app, you can simply follow these steps:
1. Install Python3 on your device. You can download it from <a href="https://www.python.org/downloads/">Here</a>.
2. Install the required packages by the following command.
```
pip install -r requirements.txt
```
3. Run the file with the name "ImageAlchemyUI.py"

## Future Updates
In upcoming releases, we are planning to introduce a significant enhancement to our application by implementing cumulative effects output. This feature will revolutionize the image processing workflow by seamlessly integrating multiple effects. Specifically, the output generated by the first applied effect will serve as the input for subsequent effects. Also, we are working now on enhancing the app's performance to suit Computationally weak devices. All that besides, new processing techniques are coming soon.

## Help

If you encounter any issues or have questions, feel free to reach out.

## Contributors

Gratitude goes out to all team members for their valuable contributions to this project.

<div align="left">
  <a href="https://github.com/cln-Kafka">
    <img src="https://avatars.githubusercontent.com/u/100665578?v=4" width="100px" alt="@Kareem Noureddine">
  </a>
  <a href="https://github.com/Nadaaomran">
    <img src="https://avatars.githubusercontent.com/u/104179154?v=4" width="100px" alt="@Nadaaomran">
  </a>
  <a href="https://github.com/joyou159">
    <img src="https://avatars.githubusercontent.com/u/85418161?v=4" width="100px" alt="@joyou159">
  </a>
  <a href="https://github.com/nouran-19">
    <img src="https://avatars.githubusercontent.com/u/99448829?v=4" width="100px" alt="@nouran-19">
  </a>
  <a href="https://github.com/MuhammadSamiAhmad">
    <img src="https://avatars.githubusercontent.com/u/101589634?v=4" width="100px" alt="@M.Sami">
  </a>
</div>

## License

All rights reserved © 2024 to Team 02 - Systems & Biomedical Engineering, Cairo University (Class 2025)
