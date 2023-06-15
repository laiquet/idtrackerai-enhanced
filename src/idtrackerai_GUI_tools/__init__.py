from functools import cache
from pathlib import Path

import numpy as np

from .GUI_main_base import GUIBase
from .widgets_utils.canvas import Canvas, CanvasMouseEvent, CanvasPainter
from .widgets_utils.custom_list import CustomList
from .widgets_utils.other_utils import (
    LabeledSlider,
    LabelRangeSlider,
    QHLine,
    WrappedLabel,
    build_ROI_patches_from_list,
    key_event_modifier,
)
from .widgets_utils.video_paths_holder import VideoPathHolder
from .widgets_utils.video_player import VideoPlayer


@cache
def get_cmap():
    parent_dir = Path(__file__).parent
    for file in parent_dir.glob("cmap_*"):
        return np.loadtxt(parent_dir / file, dtype=np.uint8)
    raise FileNotFoundError(parent_dir)


__all__ = [
    "LabelRangeSlider",
    "CustomList",
    "WrappedLabel",
    "Canvas",
    "CanvasPainter",
    "LabeledSlider",
    "GUIBase",
    "VideoPlayer",
    "VideoPathHolder",
    "key_event_modifier",
    "build_ROI_patches_from_list",
    "QHLine",
    "CanvasMouseEvent",
]
