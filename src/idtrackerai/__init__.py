from importlib import metadata

# Video has to be the first class to be imported
from idtrackerai.video import Video

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
    "Video",
    "Fragment",
]
