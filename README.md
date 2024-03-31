# Image Alchemy
[Add a brief]

![App UI](https://github.com/Computer-Vision-Spring-2024/Task-2/blob/main/README_resources/App_UI.png)


## Table of Contents:
- [Description](#description)
- [Features](#project-features)
- [Quick Preview](#quick-preview)
- [Shortcuts](#shortcuts)
- [App Structure](#app-structure)
- [Executing program](#executing-program)
- [Future Updates](#future-updates)
- [Help](#help)
- [Contributors](#contributors)
- [License](#license)

## Description

[Add a Description]

## Project Features
:white_check_mark: **3 Ways to add an image to the viewport**
- :heavy_check_mark: Using the import button in the **_File_** menu in the menu bar, or use the [shortcut](#shortcuts)
- :heavy_check_mark: Drag and drop an image from your system
- :heavy_check_mark: Pick an image from the library included in the **_Tools_** menu in the menu bar

:white_check_mark: **Digital Image Processing Techniques**
- :heavy_check_mark: Convert to grayscale
- :heavy_check_mark: Add 3 different types of noise
- :heavy_check_mark: Filter Image with 3 different types of filters
- :heavy_check_mark: 

## Quick Preview

### Browse Images
![Browse Images](README_resources/Import__gif.gif)

## Shortcuts
<center>

| Shortcut     | Functionality                                                        |
|--------------|----------------------------------------------------------------------|
| Ctrl+I       | Import Image                                                         |
| Ctrl+S       | Save the output image of the current image as PNG                    |
| Shift+Ctrl+S | Save the output image of the current image and specify the extension |
| Alt+S        | Save all the output images of the opened images                      |
| Ctrl+Q       | Exit the app                                                         |
| Ctrl+C       | Show a guide on how to use the app (Currently unavailable)           |
| Ctrl+H       | Open the documentation of the app (Currently unavailable)            |

</center>

## App Structure
<table>
  <tr>
    <td>Structure's Description: The files and our usage of OOP and how the classes communicate or managed by the backend
    </td>
    <td>

    --- Folder Structure ---
    [Classes]
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
    HelperFunctions.py
    ImageAlchemyBackend.py
    ImageAlchemyUI.py
    [Resources]
        ├── 4-B4W5KT.jpg
        ├── color.jpg
        ├── grey.jpg
        ├── [HoughSamples]
            ├── [Circle]
            ├── [Ellipse]
            └── [Line]
        ├── [Icons]
        ├── images.jpeg
        ├── snake_animation.gif
        ├── snake_animation_02.gif
        └── [Themes]
            ├── BlackTheme.qss
            ├── Stylesheet.qss
            └── WhiteTheme.qss
    TO_DO.md

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
