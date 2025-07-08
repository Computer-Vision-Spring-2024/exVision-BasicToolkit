from PyQt5.QtWidgets import QGroupBox, QPushButton, QVBoxLayout


class FaceRecognitionGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.face_recognition_effect = None

        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        # Toggle Query
        self.toggle_query = QPushButton()
        self.toggle_query.setObjectName("toggle_query")
        self.toggle_query.setText("Toggle Query")

        # Apply Face Recognition
        self.apply_face_recognition = QPushButton()
        self.apply_face_recognition.setObjectName("apply_face_recognition")
        self.apply_face_recognition.setText("Apply")

        # Show ROC
        self.show_roc = QPushButton()
        self.show_roc.setObjectName("show_roc")
        self.show_roc.setText("Show ROC")

        # Add Widgets to Main Layout
        self.main_layout.addWidget(self.toggle_query)
        self.main_layout.addWidget(self.apply_face_recognition)
        self.main_layout.addWidget(self.show_roc)

        self.setLayout(self.main_layout)
