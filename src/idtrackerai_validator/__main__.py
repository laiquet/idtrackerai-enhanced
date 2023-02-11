import sys
from argparse import ArgumentParser
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QStyleFactory

from idtrackerai.utils import initLogger
from idtrackerai_validator import ValidationGUI


def input_args():
    parser = ArgumentParser()
    parser.add_argument(
        "session_directory",
        help="Session directory to validate",
        type=Path,
        default=None,
        nargs="?",
    )
    return parser.parse_args()


def main():
    args = input_args()
    initLogger()
    app = QApplication(sys.argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    window = ValidationGUI(args.session_directory)
    window.show()
    app.exec()
