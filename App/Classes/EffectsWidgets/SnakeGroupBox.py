from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
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
        self.GAMMA_spinbox = QDoubleSpinBox()
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
        
        initial_area_layout = QHBoxLayout()
        area1_label = QLabel("Initial Area:")
        self.area1_line_edit = QLineEdit()
        self.area1_line_edit.setReadOnly(True)
        initial_area_layout.addWidget(area1_label)
        initial_area_layout.addWidget(self.area1_line_edit)
        
        initial_perimeter_layout = QHBoxLayout()
        perimeter1_label = QLabel("Initial perimeter:")
        self.perimeter1_line_edit = QLineEdit()
        self.perimeter1_line_edit.setReadOnly(True)
        initial_perimeter_layout.addWidget(perimeter1_label)
        initial_perimeter_layout.addWidget(self.perimeter1_line_edit)
        
        final_area_layout = QHBoxLayout()
        area2_label = QLabel("Final Area:")
        self.area2_line_edit = QLineEdit()
        self.area2_line_edit.setReadOnly(True)
        final_area_layout.addWidget(area2_label)
        final_area_layout.addWidget(self.area2_line_edit)
        
        final_perimeter_layout = QHBoxLayout()
        perimeter2_label = QLabel("Final Perimeter:")
        self.perimeter2_line_edit = QLineEdit()
        self.perimeter2_line_edit.setReadOnly(True)
        final_perimeter_layout.addWidget(perimeter2_label)
        final_perimeter_layout.addWidget(self.perimeter2_line_edit)
        
        self.export_button = QPushButton("Export Chain Code")
        self.export_label= QLabel("")
        self.export_label.setStyleSheet("color: red;")
        
        
        main_layout.addLayout(self.ALPHA_HLayout)
        main_layout.addLayout(self.BETA_HLayout)
        main_layout.addLayout(self.GAMMA_HLayout)
        main_layout.addLayout(self.iterations_HLayout)
        main_layout.addLayout(initial_area_layout)
        main_layout.addLayout(initial_perimeter_layout)
        main_layout.addLayout(final_area_layout)
        main_layout.addLayout(final_perimeter_layout)
        main_layout.addWidget(self.export_button)
        main_layout.addWidget(self.export_label)

        self.setLayout(main_layout)
