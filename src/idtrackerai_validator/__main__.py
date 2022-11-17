from PyQt6.QtWidgets import QApplication
from idtrackerai_validator import Window
from idtrackerai_app.themes import apply_style
from idtrackerai_app.__main__ import init_logger
import sys

# init_logger()
app = QApplication(sys.argv)
apply_style(app)
window = Window()
window.show()
app.exec()
