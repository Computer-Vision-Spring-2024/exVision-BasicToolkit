from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class NormalizeGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.normalizer_effect = None

        self.label = QLabel("Normalizer")

        self.normalizer_combo_box = QComboBox()
        self.lookup = [
            "simple rescale norm",
            "Zero Mean && Unit Variance norm",
            "Min Max Scaling norm",
            "alpha beta norm",
        ]
        self.normalizer_combo_box.addItems(self.lookup)

        hbox = QHBoxLayout()
        hbox.addWidget(self.label)
        hbox.addWidget(self.normalizer_combo_box)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(hbox)
        self.setLayout(self.vbox)

        self.alpha_layout = QHBoxLayout()
        self.alpha_label = QLabel("Alpha Value:")
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setRange(0.0, 1.0)
        self.alpha_spinbox.setObjectName("alpha_spinbox")  # Set object name

        self.beta_layout = QHBoxLayout()
        self.beta_label = QLabel("Beta Value:")
        self.beta_spinbox = QDoubleSpinBox()
        self.beta_spinbox.setRange(0.0, 1.0)
        self.beta_spinbox.setObjectName("beta_spinbox")  # Set object name

        self.update_groupbox_options(0)

    def update_groupbox_options(self, index):
        if index == 3:  # alpha beta norm
            self.alpha_layout.addWidget(self.alpha_label)
            self.alpha_layout.addWidget(self.alpha_spinbox)

            self.beta_layout.addWidget(self.beta_label)
            self.beta_layout.addWidget(self.beta_spinbox)

            self.vbox.addLayout(self.alpha_layout)
            self.vbox.addLayout(self.beta_layout)
        else:
            pass
