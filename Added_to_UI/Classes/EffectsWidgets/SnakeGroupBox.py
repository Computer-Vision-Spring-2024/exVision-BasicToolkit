from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QDoubleSpinBox,
)


class SnakeGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.snake_effect = None

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        
        self.ALPHA_label = QLabel("Alpha value")
        self.ALPHA_spinbox = QDoubleSpinBox()
        self.ALPHA_spinbox.setSingleStep(0.1)
        # Intialize a value for ALPHA
        self.ALPHA_spinbox.setValue(0.5)
        self.ALPHA_HLayout = QHBoxLayout()
        self.ALPHA_HLayout.addWidget(self.ALPHA_label)
        self.ALPHA_HLayout.addWidget(self.ALPHA_spinbox)
        
        self.BETA_label = QLabel("Beta value")
        self.BETA_spinbox = QDoubleSpinBox()
        self.BETA_spinbox.setSingleStep(0.1)
        # Intialize a value for BETA
        self.BETA_spinbox.setValue(1.0)
        self.BETA_HLayout = QHBoxLayout()
        self.BETA_HLayout.addWidget(self.BETA_label)
        self.BETA_HLayout.addWidget(self.BETA_spinbox)
        
        self.GAMMA_label = QLabel("Gamma value")
        self.GAMMA_spinbox =QDoubleSpinBox()
        self.GAMMA_spinbox.setSingleStep(0.1)
        # Intialize a value for GAMMA
        self.GAMMA_spinbox.setValue(0.2)
        self.GAMMA_HLayout = QHBoxLayout()
        self.GAMMA_HLayout.addWidget(self.GAMMA_label)
        self.GAMMA_HLayout.addWidget(self.GAMMA_spinbox)

        self.iterations_label = QLabel("Number of Iterations")
        self.iterations_spinbox = QSpinBox()
        # Intialize a value for the number of iterations
        self.iterations_spinbox.setValue(14)
        self.iterations_HLayout = QHBoxLayout()
        self.iterations_HLayout.addWidget(self.iterations_label)
        self.iterations_HLayout.addWidget(self.iterations_spinbox)
        main_layout.addLayout(self.ALPHA_HLayout)
        main_layout.addLayout(self.BETA_HLayout)
        main_layout.addLayout(self.GAMMA_HLayout)
        main_layout.addLayout(self.iterations_HLayout)

        self.setLayout(main_layout)
