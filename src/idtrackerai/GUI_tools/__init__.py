from functools import cache
from pathlib import Path

import numpy as np

from .GUI_main_base import GUIBase
from .widgets_utils.canvas import Canvas, CanvasMouseEvent, CanvasPainter
from .widgets_utils.custom_list import CustomList
from .widgets_utils.other_utils import (
    LightPopUp,
    QHLine,
    WrappedLabel,
    build_ROI_patches_from_list,
    get_path_from_points,
    key_event_modifier,
)
from .widgets_utils.sliders import InvertibleSlider, LabelRangeSlider
from .widgets_utils.video_paths_holder import VideoPathHolder
from .widgets_utils.video_player import VideoPlayer


@cache
def get_cmap():
    parent_dir = Path(__file__).parent
    for file in parent_dir.glob("cmap_*"):
        return np.loadtxt(parent_dir / file, dtype=np.uint8)
    raise FileNotFoundError(parent_dir)


point_colors: list[int] = [
    0x9467BD,
    0x2CA02C,
    0xBCBD22,
    0xFF7F0E,
    0x8C564B,
    0xE377C2,
    0x7F7F7F,
    0x17BECF,
]


__all__ = [
    "LabelRangeSlider",
    "CustomList",
    "WrappedLabel",
    "Canvas",
    "CanvasPainter",
    "GUIBase",
    "VideoPlayer",
    "VideoPathHolder",
    "key_event_modifier",
    "build_ROI_patches_from_list",
    "QHLine",
    "CanvasMouseEvent",
    "get_path_from_points",
    "LightPopUp",
    "InvertibleSlider",
]
