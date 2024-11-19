from .animals_detection import animals_detection_API
from .segmentation import (
    generate_background_from_frame_stack,
    generate_frame_stack,
    process_frame,
    to_gray_scale,
)

__all__ = [
    "animals_detection_API",
    "generate_background_from_frame_stack",
    "generate_frame_stack",
    "process_frame",
    "to_gray_scale",
]
