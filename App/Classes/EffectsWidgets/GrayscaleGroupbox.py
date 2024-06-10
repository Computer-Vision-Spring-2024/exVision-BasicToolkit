from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QSlider


class GrayscaleConverter(QGroupBox):
    def __init__(self):
        super().__init__("Grayscale Converter")
        self.initUI()

    def initUI(self):
        # Creating labels
        label_red = QLabel("Red")
        label_green = QLabel("Green")
        label_blue = QLabel("Blue")

        # Creating sliders
        slider_red = QSlider()
        slider_green = QSlider()
        slider_blue = QSlider()
        slider_red.setOrientation(1)
        slider_green.setOrientation(1)
        slider_blue.setOrientation(1)
        slider_red.setRange(0, 255)
        slider_green.setRange(0, 255)
        slider_blue.setRange(0, 255)

        # Creating layout
        main_layout = QGridLayout()

        # Add widgets to layout
        main_layout.addWidget(label_red, 0, 0)
        main_layout.addWidget(label_green, 1, 0)
        main_layout.addWidget(label_blue, 2, 0)
        main_layout.addWidget(slider_red, 0, 1)
        main_layout.addWidget(slider_green, 1, 1)
        main_layout.addWidget(slider_blue, 2, 1)

        # Set layout to group box
        self.setLayout(main_layout)
