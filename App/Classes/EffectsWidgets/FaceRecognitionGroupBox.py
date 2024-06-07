from PyQt5.QtWidgets import QGroupBox, QPushButton, QVBoxLayout


class FaceRecognition(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.title = title
        self.segmentation_effect = None

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

        # Add Widgets to Main Layout
        self.main_layout.addWidget(self.toggle_query)
        self.main_layout.addWidget(self.apply_face_recognition)

        self.setLayout(self.main_layout)
