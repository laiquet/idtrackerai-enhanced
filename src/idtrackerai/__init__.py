from contextlib import suppress
from importlib import metadata

with suppress(ImportError):
    # PyQt has to be imported before CV2 (importing idtrackerai stuff implies CV2)
    # If not, the QFileDialog.getFileNames() does not load the icons, very weird
    from qtpy.QtWidgets import QApplication  # noqa F401

# Video has to be the first class to be imported
from idtrackerai.session import Session

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
]
