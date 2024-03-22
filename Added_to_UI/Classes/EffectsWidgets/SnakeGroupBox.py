from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class SnakeGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.snake_effect = None

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        # Frequency filter Type
        filter_type_layout = QHBoxLayout()
        self.filter_type_label = QLabel("Filter Type:")
        self.filter_type_comb = QComboBox()
        self.filter_type_comb.setCurrentIndex(0)
        self.filter_type_comb.addItems(["Low-Pass filter", "High-Pass filter"])
        filter_type_layout.addWidget(self.filter_type_label)
        filter_type_layout.addWidget(self.filter_type_comb)

        self.cutoff_label = QLabel("Cutoff Frequency")
        self.cutoff_spinbox = QSpinBox()
        # Intialize a value for the cutoff frequency
        self.cutoff_spinbox.setValue(20)
        self.cutoff_HLayout = QHBoxLayout()
        self.cutoff_HLayout.addWidget(self.cutoff_label)
        self.cutoff_HLayout.addWidget(self.cutoff_spinbox)

        main_layout.addLayout(filter_type_layout)
        main_layout.addLayout(self.cutoff_HLayout)

        self.setLayout(main_layout)
