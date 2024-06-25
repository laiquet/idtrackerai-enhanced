from colorsys import hsv_to_rgb
from typing import Any, Iterable

import numpy as np

from .GUI_main_base import GUIBase
from .widgets_utils.canvas import Canvas, CanvasMouseEvent, CanvasPainter
from .widgets_utils.custom_list import CustomList
from .widgets_utils.other_utils import (
    LightPopUp,
    QHLine,
    TransparentDisabledOverlay,
    WrappedLabel,
    build_ROI_patches_from_list,
    get_path_from_points,
    key_event_modifier,
)
from .widgets_utils.sliders import InvertibleSlider, LabelRangeSlider
from .widgets_utils.video_paths_holder import VideoPathHolder
from .widgets_utils.video_player import VideoPlayer


def get_cmap(
    values: np.ndarray | Iterable[float] | int,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if isinstance(values, int):
        rgb_list = [hsv_to_rgb(h / values, 1, 1) for h in range(values)]
    else:
        rgb_list = [hsv_to_rgb(h, 1, 1) for h in values]

    return (np.asarray(rgb_list) * 255).astype(np.uint8)


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
    "TransparentDisabledOverlay",
]
