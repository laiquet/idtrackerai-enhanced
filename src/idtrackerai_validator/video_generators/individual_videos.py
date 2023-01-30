import logging

import cv2
import numpy as np
from rich.progress import track

from idtrackerai import Video
from idtrackerai_GUI_tools import VideoPathHolder


def draw_individual_frame(
    frame: np.ndarray,
    draw_in_gray: bool,
    ordered_centroid: np.ndarray,
    positions: list[tuple[int, int]],
    width: int,
    height: int,
    size: int,
    labels: list[str],
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    size2 = size // 2
    if draw_in_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for cur_id, (x, y) in enumerate(ordered_centroid):
        draw_x, draw_y = positions[cur_id]
        canvas = cv2.putText(
            canvas,
            labels[cur_id],
            (draw_x, draw_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
        if x > 0 and y > 0:
            if draw_in_gray:
                mini_frame = frame[
                    max(0, y - size2) : y + size2, max(0, x - size2) : x + size2, None
                ]
            else:
                mini_frame = frame[
                    max(0, y - size2) : y + size2, max(0, x - size2) : x + size2
                ]
            canvas[
                draw_y : draw_y + mini_frame.shape[0],
                draw_x : draw_x + mini_frame.shape[1],
            ] = mini_frame
    return canvas


def generate_individual_video(
    video: Video,
    trajectories: np.ndarray,
    draw_in_gray: bool,
    starting_frame: int,
    ending_frame: int | None,
    **kargs,
):
    draw_in_gray = draw_in_gray
    if draw_in_gray:
        logging.info(f"Drawing original video in grayscale")

    trajectories = trajectories.astype(int)

    path_to_save_video = video.session_folder / (
        video.video_paths[0].stem + "_individuals.avi"
    )

    n_rows = int(np.sqrt(video.number_of_animals))
    n_cols = int(video.number_of_animals / n_rows - 0.0001) + 1

    miniframe_size = 2 * (int(video.median_body_length_full_resolution) // 2)
    extra_lower_pad = 10
    bbox_side_pad = 10
    bbox_top_pad = 30
    full_bbox_width = miniframe_size + 2 * bbox_side_pad
    out_video_width = n_cols * full_bbox_width

    full_bbox_height = miniframe_size + bbox_top_pad
    out_video_height = n_rows * full_bbox_height + extra_lower_pad

    positions = [
        (
            full_bbox_width * (i % n_cols) + bbox_side_pad,
            full_bbox_height * (i // n_cols) + bbox_top_pad,
        )
        for i in range(video.number_of_animals)
    ]

    labels = video.identities_labels

    videoPathHolder = VideoPathHolder(video.video_paths)

    ending_frame = len(trajectories) - 1 if ending_frame is None else ending_frame
    logging.info(f"Drawing from frame {starting_frame} to {ending_frame}")

    video_writer = cv2.VideoWriter(
        str(path_to_save_video),
        cv2.VideoWriter_fourcc(*"XVID"),
        video.frames_per_second,
        (out_video_width, out_video_height),
    )
    for frame in track(
        range(starting_frame, ending_frame), description="Rendering video:"
    ):
        img = videoPathHolder.frameColor(frame)

        drown_frame = draw_individual_frame(
            img,
            draw_in_gray,
            trajectories[frame],
            positions,
            width=out_video_width,
            height=out_video_height,
            size=miniframe_size,
            labels=labels,
        )

        video_writer.write(drown_frame)

    logging.info(f"Video generated in {path_to_save_video}")
