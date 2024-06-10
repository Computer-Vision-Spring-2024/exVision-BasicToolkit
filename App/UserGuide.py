import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QMovie, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
)


class UserGuideDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("User Guide")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        # Add an icon
        icon = QIcon()
        icon.addPixmap(
            QPixmap("Resources/Icons/App_Icon.png"),
            QIcon.Normal,
            QIcon.Off,
        )
        self.setWindowIcon(icon)

        self.gif_index = 0
        self.gifs = [
            "C:/Users/Kareem/Desktop/ImageAlchemy/README_resources/Import__gif.gif",
            "C:/Users/Kareem/Desktop/ImageAlchemy/README_resources/Navigate.gif",
            "C:/Users/Kareem/Desktop/ImageAlchemy/README_resources/noise.gif",
        ]  # Paths to your GIFs
        self.total_gifs = len(self.gifs)

        self.gif_viewer = QLabel()
        self.gif_viewer.setAlignment(Qt.AlignCenter)
        self.update_gif()

        self.back_button = QPushButton("Back", clicked=self.show_previous_gif)
        self.next_button = QPushButton("Next", clicked=self.show_next_gif)

        self.main_layout = QVBoxLayout()
        spacerItem_left = QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum
        )
        spacerItem_right = QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum
        )

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addItem(spacerItem_left)
        self.buttons_layout.addWidget(self.back_button)
        self.buttons_layout.addWidget(self.next_button)
        self.buttons_layout.addItem(spacerItem_right)

        self.main_layout.addWidget(self.gif_viewer)
        self.main_layout.addLayout(self.buttons_layout)

        self.setLayout(self.main_layout)
        self.update_button_states()

    def update_gif(self):
        if self.gif_index < self.total_gifs:
            gif_path = self.gifs[self.gif_index]
            movie = QMovie(gif_path)
            self.gif_viewer.setMovie(movie)
            movie.start()

    def show_previous_gif(self):
        if self.gif_index > 0:
            self.gif_index -= 1
            self.update_gif()
            self.update_button_states()

    def show_next_gif(self):
        if self.gif_index < self.total_gifs - 1:
            self.gif_index += 1
            self.update_gif()
            self.update_button_states()
        elif self.gif_index == self.total_gifs - 1:
            self.next_button.setText("Finished")
            self.next_button.setEnabled(False)

    def update_button_states(self):
        self.back_button.setEnabled(self.gif_index > 0)
        self.next_button.setEnabled(self.gif_index < self.total_gifs - 1)
