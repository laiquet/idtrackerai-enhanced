"""isort:skip_file"""

import multiprocessing
import os
import sys
from contextlib import suppress
from importlib import metadata

if "forkserver" in multiprocessing.get_all_start_methods():
    multiprocessing.set_start_method("forkserver", force=True)

with suppress(ImportError):
    # PyQt has to be imported before CV2 (importing idtrackerai stuff implies CV2)
    # If not, the QFileDialog.getFileNames() does not load the icons, very weird

    # lets try to go for PyQt6
    if "QT_API" not in os.environ:
        os.environ["QT_API"] = "pyqt6"
    try:
        from qtpy.QtWidgets import QApplication  # noqa F401
    except ImportError:
        # ups! PyQt6 failed. Lets forget everything and try again
        os.environ.pop("QT_API")
        # in Ubuntu I've seen PyQt6 to remain in sys.modules even though it failed importing
        sys.modules.pop("PyQt6", None)
        from qtpy.QtWidgets import QApplication  # noqa F401

from .utils import IdtrackeraiError, conf

# Video has to be the first class to be imported
from .session import Session

from .blob import Blob
from .fragment import Fragment
from .globalfragment import GlobalFragment
from .list_of_blobs import ListOfBlobs
from .list_of_fragments import ListOfFragments
from .list_of_global_fragments import ListOfGlobalFragments

__version__ = metadata.version("idtrackerai")


__all__ = [
    "Blob",
    "ListOfBlobs",
    "ListOfFragments",
    "ListOfGlobalFragments",
    "ListOfGlobalFragments",
    "GlobalFragment",
    "Session",
    "Fragment",
    "IdtrackeraiError",
    "conf",
]
