from PyQt5 import QtCore, QtGui, QtWidgets


class ThemeDialog(QtWidgets.QDialog):
    stylesheetSelected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Theme")
        self.resize(300, 150)

        self.theme_buttons = []

        mainLayout = QtWidgets.QVBoxLayout()

        layout = QtWidgets.QHBoxLayout()

        themes = [
            ("White Theme", "White", "Resources/WhiteTheme.qss"),
            ("Black Theme", "Black", "Resources/BlackTheme.qss"),
            ("Blue Theme", "Blue", "Resources/BlueTheme.qss"),
        ]

        for theme_name, color, stylesheet_path in themes:
            button = QtWidgets.QPushButton()
            button.setToolTip(theme_name)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setStyleSheet("QPushButton:checked { border: 4px solid orange;}")
            style = f"background-color: {color}; color: white;"
            button.setStyleSheet(style)
            button.setCheckable(True)
            button.setMinimumHeight(50)
            button.setMaximumWidth(50)
            button.clicked.connect(
                lambda _, path=stylesheet_path: self.apply_theme(path)
            )
            layout.addWidget(button)
            self.theme_buttons.append(button)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(button_box)

        mainLayout.addLayout(layout)
        mainLayout.addLayout(layout2)
        self.setLayout(mainLayout)

    def apply_theme(self, stylesheet_path):
        self.stylesheetSelected.emit(stylesheet_path)
