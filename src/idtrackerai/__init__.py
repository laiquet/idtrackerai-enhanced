from importlib import metadata
from warnings import warn

# Video has to be the first class to be imported
from idtrackerai.session import Session

from .blob import Blob
from .fragment import Fragment
from .globalfragment import GlobalFragment
from .list_of_blobs import ListOfBlobs
from .list_of_fragments import ListOfFragments
from .list_of_global_fragments import ListOfGlobalFragments

__version__ = metadata.version("idtrackerai")


class Video(Session):
    "Backward compatibility since the rename of the `Video` class for `Session`"

    def __new__(cls):
        warn("Video is deprecated since v5.2.3, it has been renamed to `Session`")
        return super().__new__(cls)


__all__ = [
    "Blob",
    "ListOfBlobs",
    "ListOfFragments",
    "ListOfGlobalFragments",
    "ListOfGlobalFragments",
    "GlobalFragment",
    "Session",
    "Fragment",
    "Video",
]
