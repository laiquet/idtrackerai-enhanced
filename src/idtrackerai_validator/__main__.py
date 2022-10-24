from PyQt6.QtWidgets import QApplication
from idtrackerai_validator import Window
from idtrackerai_app.themes import apply_style
import sys
import logging

app = QApplication(sys.argv)
apply_style(app)
window = Window()
window.show()
app.exec()
