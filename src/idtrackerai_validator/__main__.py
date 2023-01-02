import sys
from argparse import ArgumentParser
from pathlib import Path

from idtrackerai_app.__main__ import init_logger
from idtrackerai_validator import ValidationGUI
from PyQt6.QtWidgets import QApplication


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
    init_logger()
    app = QApplication(sys.argv)
    window = ValidationGUI(args.session_directory)
    window.show()
    app.exec()
