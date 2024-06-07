from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class HistogramGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setTitle(title)
        self.equalizer_effect = None

        self.label = QLabel("Histogram Channel")

        self.histogram_channel = QComboBox()
        self.lookup = [
            "Histogram",
            "Cumulative Distribution",
            "Channel Histogram && Cumulative Distribution",
        ]
        self.histogram_channel.addItems(self.lookup)

        hbox = QHBoxLayout()
        hbox.addWidget(self.label)
        hbox.addWidget(self.histogram_channel)

        self.apply_button = QPushButton("Apply")
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.apply_button)

        self.setLayout(vbox)
