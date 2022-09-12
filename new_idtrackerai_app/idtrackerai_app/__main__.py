import sys, os
from PyQt6.QtWidgets import QApplication

sys.path.append(os.getcwd())


def start():
    import logging
    from rich.logging import RichHandler
    from rich.console import Console

    logger_width_when_no_terminal = 150
    try:
        os.get_terminal_size()
    except OSError:
        # stdout is sent to file. We define logger width to a constant
        size = logger_width_when_no_terminal
    else:
        # stdout is sent to terminal
        # We define logger width to adapt to the terminal width
        size = None

    # The first handler is the terminal, the second one the .log file,
    # both rendered with Rich and full logging (level=0)
    logging.basicConfig(
        level=0,
        format="%(message)s",
        datefmt="%b %d %H:%M:%S",
        handlers=[
            RichHandler(console=Console(width=size)),
            RichHandler(
                console=Console(
                    file=open("idtrackerai-app.log", "w"),
                    width=logger_width_when_no_terminal,
                ),
            ),
        ],
    )

    logger = logging.getLogger()
    logger.info("Welcome to idtracker.ai")
    # from pyforms import start_app
    from confapp import conf

    try:
        import local_settings

        # print(conf.PYFORMS_MODE)
        conf += local_settings
    except ImportError:
        logger.info("Local settings file not available.")
    import idtrackerai

    conf += idtrackerai.constants

    logging.getLogger("PyQt5").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    from .GUI_main import Window

    app = QApplication(sys.argv)

    window = Window()

    window.show()
    app.exec()

    # try:
    #     start_app(App, geometry=(100, 100, 800, 600))
    # except SystemExit:
    #     pass
    # except Exception as e:
    #     logger.info(e, exc_info=True)
    #     import traceback

    #     ex_type, ex, tb = sys.exc_info()
    #     traceback.print_exception(ex_type, ex, tb)


# Execute the application
if __name__ == "__main__":
    start()
