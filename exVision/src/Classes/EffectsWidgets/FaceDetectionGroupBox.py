from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class FaceDetectionGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.face_detection_effect = None

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        # Last Stage Threshold
        self.detection_HBoxLayout = QHBoxLayout()
        self.detection_HBoxLayout.setObjectName("detection_HBoxLayout")

        self.last_stage_threshold_label = QLabel()
        self.last_stage_threshold_label.setObjectName("last_stage_threshold_label")
        self.last_stage_threshold_label.setText("Last Stage Threshold")

        self.last_stage_threshold_spinbox = QDoubleSpinBox()
        self.last_stage_threshold_spinbox.setObjectName("last_stage_threshold_spinbox")
        self.last_stage_threshold_spinbox.setValue(1)
        self.last_stage_threshold_spinbox.setSingleStep(0.1)
        self.last_stage_threshold_spinbox.setMinimum(0)
        self.last_stage_threshold_spinbox.setMaximum(9)

        self.detection_HBoxLayout.addWidget(self.last_stage_threshold_label)
        self.detection_HBoxLayout.addWidget(self.last_stage_threshold_spinbox)

        self.apply_face_detection = QPushButton()
        self.apply_face_detection.setObjectName("apply_face_detection")
        self.apply_face_detection.setText("Apply Face Detection")

        # Add Widgets to Main Layout
        self.main_layout.addLayout(self.detection_HBoxLayout)
        self.main_layout.addWidget(self.apply_face_detection)

        self.setLayout(self.main_layout)
