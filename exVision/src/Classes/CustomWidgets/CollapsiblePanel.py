from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CollapsiblePanel(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(
        self,
        parent=None,
        header_icon="Resources/Icons/RightArrows.png",
        header_text="",
        collapsed_width=30,
        expanded_width=200,
    ):
        super().__init__(parent)
        self._collapsed_width = collapsed_width
        self._expanded_width = expanded_width
        self._is_expanded = False

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header
        self.header = QFrame(self)
        self.header.setFrameShape(QFrame.StyledPanel)
        self.header.setFrameShadow(QFrame.Raised)

        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.header_btn = QPushButton(self.header)
        self.header_btn.setCheckable(True)
        self.header_btn.setChecked(False)
        if header_icon:
            self.header_btn.setIcon(QIcon(header_icon))
        self.header_btn.setText(header_text)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_layout.addWidget(self.header_btn)
        self.main_layout.addWidget(self.header)

        # Collapsed and Expanded Views
        self.collapsed_widget = QWidget(self)
        self.collapsed_widget.setMinimumWidth(self._collapsed_width)
        self.collapsed_widget.setMaximumWidth(self._collapsed_width)
        self.collapsed_layout = QVBoxLayout(self.collapsed_widget)
        self.collapsed_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.collapsed_widget)

        self.expanded_widget = QWidget(self)
        self.expanded_widget.setMinimumWidth(self._expanded_width)
        self.expanded_widget.setMaximumWidth(self._expanded_width)
        self.expanded_layout = QVBoxLayout(self.expanded_widget)
        self.expanded_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.expanded_widget)

        self.setFocusPolicy(Qt.NoFocus)
        self.set_expanded(False)

        self.header_btn.toggled.connect(self.set_expanded)
        self.header_btn.toggled.connect(self.toggled.emit)

    def set_expanded(self, expanded: bool):
        self._is_expanded = expanded
        self.expanded_widget.setVisible(expanded)
        self.collapsed_widget.setVisible(not expanded)
        if expanded:
            self.header_btn.setIcon(QIcon("Resources/Icons/LeftArrows.png"))
            self.setMinimumWidth(self._expanded_width)
            self.setMaximumWidth(self._expanded_width)
        else:
            self.header_btn.setIcon(QIcon("Resources/Icons/RightArrows.png"))
            self.setMinimumWidth(self._collapsed_width)
            self.setMaximumWidth(self._collapsed_width)

    def add_to_collapsed(self, widget):
        self.collapsed_layout.addWidget(widget)

    def add_to_expanded(self, widget):
        self.expanded_layout.addWidget(widget)

    def is_expanded(self):
        return self._is_expanded
