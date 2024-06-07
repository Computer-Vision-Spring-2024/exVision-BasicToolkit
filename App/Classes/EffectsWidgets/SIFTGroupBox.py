from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)


class SIFTGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.sift_effect = None

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        # n_octaves
        self.n_octaves_HBoxLayout = QHBoxLayout()
        self.n_octaves_HBoxLayout.setObjectName("n_octaves_HBoxLayout")

        self.n_octaves_label = QLabel()
        self.n_octaves_label.setObjectName("n_octaves")
        self.n_octaves_label.setText("n_octaves")

        self.n_octaves_spin_box = QSpinBox()
        self.n_octaves_spin_box.setObjectName("n_octaves_spin_box")
        self.n_octaves_spin_box.setValue(4)
        self.n_octaves_spin_box.setSingleStep(1)
        self.n_octaves_spin_box.setMinimum(4)
        self.n_octaves_spin_box.setMaximum(8)

        self.n_octaves_HBoxLayout.addWidget(self.n_octaves_label)
        self.n_octaves_HBoxLayout.addWidget(self.n_octaves_spin_box)

        # s_value
        self.s_value_HBoxLayout = QHBoxLayout()
        self.s_value_HBoxLayout.setObjectName("s_value_HBoxLayout")

        self.s_value_label = QLabel()
        self.s_value_label.setObjectName("s_value")
        self.s_value_label.setText("s_value")

        self.s_value_spin_box = QSpinBox()
        self.s_value_spin_box.setObjectName("s_value_spin_box")
        self.s_value_spin_box.setValue(2)
        self.s_value_spin_box.setSingleStep(1)
        self.s_value_spin_box.setMinimum(2)
        self.s_value_spin_box.setMaximum(5)

        self.s_value_HBoxLayout.addWidget(self.s_value_label)
        self.s_value_HBoxLayout.addWidget(self.s_value_spin_box)

        # sigma_base
        self.sigma_base_HBoxLayout = QHBoxLayout()
        self.sigma_base_HBoxLayout.setObjectName("sigma_base_HBoxLayout")

        self.sigma_base_label = QLabel()
        self.sigma_base_label.setObjectName("sigma_base")
        self.sigma_base_label.setText("sigma_base")

        self.sigma_base_spin_box = QDoubleSpinBox()
        self.sigma_base_spin_box.setObjectName("sigma_base_spin_box")
        self.sigma_base_spin_box.setValue(1.6)
        self.sigma_base_spin_box.setSingleStep(0.1)
        self.sigma_base_spin_box.setMinimum(1.6)
        self.sigma_base_spin_box.setMaximum(3)

        self.sigma_base_HBoxLayout.addWidget(self.sigma_base_label)
        self.sigma_base_HBoxLayout.addWidget(self.sigma_base_spin_box)

        # r_ratio
        self.r_ratio_HBoxLayout = QHBoxLayout()
        self.r_ratio_HBoxLayout.setObjectName("r_ratio_HBoxLayout")

        self.r_ratio_label = QLabel()
        self.r_ratio_label.setObjectName("r_ratio")
        self.r_ratio_label.setText("r_ratio")

        self.r_ratio_spin_box = QDoubleSpinBox()
        self.r_ratio_spin_box.setObjectName("r_ratio_spin_box")
        self.r_ratio_spin_box.setValue(10)
        self.r_ratio_spin_box.setSingleStep(1)
        self.r_ratio_spin_box.setMinimum(6)
        self.r_ratio_spin_box.setMaximum(12)

        self.r_ratio_HBoxLayout.addWidget(self.r_ratio_label)
        self.r_ratio_HBoxLayout.addWidget(self.r_ratio_spin_box)

        # contrast_th
        self.contrast_th_VBoxLayout = QVBoxLayout()
        self.contrast_th_VBoxLayout.setObjectName("contrast_th_VBoxLayout")

        self.contrast_th_label = QLabel()
        self.contrast_th_label.setObjectName("contrast_th")
        self.contrast_th_label.setText("contrast_th")

        self.contrast_th_slider = QSlider()
        self.contrast_th_slider.setOrientation(Qt.Horizontal)
        self.contrast_th_slider.setObjectName("contrast_th_slider")
        self.contrast_th_slider.setValue(30)
        self.contrast_th_slider.setSingleStep(10)
        self.contrast_th_slider.setMinimum(1)
        self.contrast_th_slider.setMaximum(100)

        self.contrast_th_VBoxLayout.addWidget(self.contrast_th_label)
        self.contrast_th_VBoxLayout.addWidget(self.contrast_th_slider)

        # confusion factor
        self.confusion_factor_VBoxLayout = QVBoxLayout()
        self.confusion_factor_VBoxLayout.setObjectName("confusion_factor_VBoxLayout")

        self.confusion_factor_label = QLabel()
        self.confusion_factor_label.setObjectName("confusion_factor")
        self.confusion_factor_label.setText("confusion_factor")

        self.confusion_factor_slider = QSlider()
        self.confusion_factor_slider.setOrientation(Qt.Horizontal)
        self.confusion_factor_slider.setObjectName("confusion_factor_slider")
        self.confusion_factor_slider.setValue(3)
        self.confusion_factor_slider.setSingleStep(1)
        self.confusion_factor_slider.setMinimum(1)
        self.confusion_factor_slider.setMaximum(10)

        self.confusion_factor_VBoxLayout.addWidget(self.confusion_factor_label)
        self.confusion_factor_VBoxLayout.addWidget(self.confusion_factor_slider)

        # Apply button and the time label
        self.apply_sift_VBoxLayout = QVBoxLayout()
        self.apply_sift_VBoxLayout.setObjectName("apply_sift_VBoxLayout")

        self.sift_elapsed_time_label = QLabel()
        self.sift_elapsed_time_label.setText("Elapsed Time is ")
        self.sift_elapsed_time_label.setMinimumSize(QSize(0, 25))
        self.sift_elapsed_time_label.setMaximumSize(QSize(16777215, 25))
        self.sift_elapsed_time_label.setObjectName("sift_elapsed_time_label")

        self.apply_sift_btn = QPushButton()
        self.apply_sift_btn.setObjectName("apply_sift_btn")
        self.apply_sift_btn.setText("Apply SIFT")

        self.apply_sift_VBoxLayout.addWidget(self.apply_sift_btn)
        self.apply_sift_VBoxLayout.addWidget(self.sift_elapsed_time_label)

        # Add all the layouts to the main layout
        self.main_layout.addLayout(self.n_octaves_HBoxLayout)
        self.main_layout.addLayout(self.s_value_HBoxLayout)
        self.main_layout.addLayout(self.sigma_base_HBoxLayout)
        self.main_layout.addLayout(self.r_ratio_HBoxLayout)
        self.main_layout.addLayout(self.contrast_th_VBoxLayout)
        self.main_layout.addLayout(self.confusion_factor_VBoxLayout)
        self.main_layout.addLayout(self.apply_sift_VBoxLayout)

        self.setLayout(self.main_layout)
