from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QSlider, QVBoxLayout


class CornerDetectionGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.segmentation_effect = None

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        # Threshold
        self.corner_detection_threshold_label = QLabel()
        self.corner_detection_threshold_label.setObjectName(
            "corner_detection_threshold_label"
        )
        self.corner_detection_threshold_label.setText("Threshold")

        self.corner_detection_threshold_slider = QSlider()
        self.corner_detection_threshold_slider.setOrientation(Qt.Horizontal)
        self.corner_detection_threshold_slider.setObjectName(
            "corner_detection_threshold_slider"
        )

        # Elapsed Time
        self.elapsed_time_label = QLabel()
        self.elapsed_time_label.setObjectName("elasped_time_label")
        self.elapsed_time_label.setText("Elapsed Time is")

        # Apply Harris
        self.apply_harris = QPushButton()
        self.apply_harris.setObjectName("apply_harris")
        self.apply_harris.setText("Reset Harris")

        # Apply Lambda Minus
        self.apply_lambda_minus = QPushButton()
        self.apply_lambda_minus.setObjectName("apply_lambda_minus")
        self.apply_lambda_minus.setText("Apply Lambda Minus")

        # Add Widgets to Layout: Region Growing
        self.main_layout.addWidget(self.corner_detection_threshold_label)
        self.main_layout.addWidget(self.corner_detection_threshold_slider)
        self.main_layout.addWidget(self.elapsed_time_label)
        self.main_layout.addWidget(self.apply_harris)
        self.main_layout.addWidget(self.apply_lambda_minus)

        self.setLayout(self.main_layout)
