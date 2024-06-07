from PyQt5.QtWidgets import QSpinBox


class OddSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.editingFinished.connect(self.adjustValue)

    def adjustValue(self):
        if self.value() % 2 == 0:
            self.setValue(self.value() + 1)
