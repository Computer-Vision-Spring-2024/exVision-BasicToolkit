import sys

import qdarktheme
from PyQt5.QtWidgets import QApplication, QMainWindow

from exVision_Backend import exVisionBackend
from exVision_Ui import ExVisionUI

# def load_stylesheet(path):
#     file = QFile(path)
#     if file.open(QFile.ReadOnly | QFile.Text):
#         stream = QTextStream(file)
#         return stream.readAll()
#     return ""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = ExVisionUI()
    ui.setupUi(main_window)
    backend = exVisionBackend(ui)
    main_window.show()

    # Load and apply the stylesheet
    # qss = load_stylesheet("Resources/Themes/Stylesheet.qss")
    # app.setStyleSheet(qss)

    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())
