from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)


class AdvancedThresholdingGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.advanced_thresholding_effect = None

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        ## Thresholding
        self.thresholding_type_HBoxLayout = QHBoxLayout()
        self.thresholding_type_HBoxLayout.setObjectName("thresholding_type_HBoxLayout")

        self.thresholding_type_label = QLabel("Thresholding")
        self.thresholding_type_label.setObjectName("thresholding_type")
        self.thresholding_type_label.setText("Thresholding Type")

        self.thresholding_type_comboBox = QComboBox()
        self.thresholding_type_comboBox.addItem("Optimal - Binary")
        self.thresholding_type_comboBox.addItem("OTSU")
        self.thresholding_type_comboBox.setObjectName("thresholding_comboBox")

        self.thresholding_type_HBoxLayout.addWidget(self.thresholding_type_label)
        self.thresholding_type_HBoxLayout.addWidget(self.thresholding_type_comboBox)

        # Local Checkbox
        self.local_checkbox = QCheckBox()
        self.local_checkbox.setObjectName("local_checkbox")
        self.local_checkbox.setChecked(False)
        self.local_checkbox.setText("Local Thresholding")

        # Global Checkbox
        self.global_checkbox = QCheckBox()
        self.global_checkbox.setObjectName("global_checkbox")
        self.global_checkbox.setChecked(True)
        self.global_checkbox.setText("Global Thresholding")

        # Local or Global
        self.local_or_global_HBoxLayout = QHBoxLayout()
        self.local_or_global_HBoxLayout.setObjectName("local_or_global_HBoxLayout")
        self.local_or_global_HBoxLayout.addWidget(self.local_checkbox)
        self.local_or_global_HBoxLayout.addWidget(self.global_checkbox)

        # Separability Measure
        self.separability_measure_label = QLabel()
        self.separability_measure_label.setObjectName("separability_measure_label")
        self.separability_measure_label.setText("Separability Measure = ")

        # OTSU Step
        self.otsu_step_HBoxLayout = QHBoxLayout()
        self.otsu_step_HBoxLayout.setObjectName("thresholding_type_HBoxLayout")

        self.otsu_step_label = QLabel()
        self.otsu_step_label.setObjectName("otsu_step_label")
        self.otsu_step_label.setText("OTSU Step")

        self.otsu_step_spinbox = QSpinBox()
        self.otsu_step_spinbox.setObjectName("otsu_step_spinbox")
        self.otsu_step_spinbox.setValue(1)
        self.otsu_step_spinbox.setSingleStep(1)
        self.otsu_step_spinbox.setMinimum(1)
        self.otsu_step_spinbox.setMaximum(200)

        self.otsu_step_HBoxLayout.addWidget(self.otsu_step_label)
        self.otsu_step_HBoxLayout.addWidget(self.otsu_step_spinbox)

        # Number of Thresholds
        self.num_of_thresholds_HBoxLayout = QHBoxLayout()
        self.num_of_thresholds_HBoxLayout.setObjectName("thresholding_type_HBoxLayout")

        self.number_of_thresholds_label = QLabel()
        self.number_of_thresholds_label.setObjectName("number_of_modes")
        self.number_of_thresholds_label.setText("Number of Thresholds = 1")

        self.number_of_thresholds_slider = QSlider()
        self.number_of_thresholds_slider.setOrientation(Qt.Horizontal)
        self.number_of_thresholds_slider.setObjectName("horizontalSlider")
        self.number_of_thresholds_slider.setValue(1)
        self.number_of_thresholds_slider.setSingleStep(1)
        self.number_of_thresholds_slider.setMinimum(1)
        self.number_of_thresholds_slider.setMaximum(5)

        self.num_of_thresholds_HBoxLayout.addWidget(self.number_of_thresholds_label)
        self.num_of_thresholds_HBoxLayout.addWidget(self.number_of_thresholds_slider)

        # Apply Thresholding
        self.apply_thresholding = QPushButton()
        self.apply_thresholding.setObjectName("apply_thresholding")
        self.apply_thresholding.setText("Apply Thresholding")

        # Add Widgets to Main Layout
        self.main_layout.addLayout(self.thresholding_type_HBoxLayout)
        self.main_layout.addLayout(self.local_or_global_HBoxLayout)
        self.main_layout.addWidget(self.separability_measure_label)
        self.main_layout.addLayout(self.otsu_step_HBoxLayout)
        self.main_layout.addLayout(self.num_of_thresholds_HBoxLayout)
        self.main_layout.addWidget(self.apply_thresholding)

        self.setLayout(self.main_layout)
