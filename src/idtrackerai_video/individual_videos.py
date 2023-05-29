import logging

import cv2
import numpy as np

from idtrackerai import Video
from idtrackerai.utils import create_dir, track
from idtrackerai_GUI_tools import VideoPathHolder


# TODO INDIVIDUAL_VIDEO_WIDTH_HEIGHT parameter
def draw_general_frame(
    positions: list[tuple[int, int]],
    size: int,
    miniframes: np.ndarray,
    canvas: np.ndarray,
):
    for cur_id in range(miniframes.shape[0]):
        draw_x, draw_y = positions[cur_id]
        canvas[draw_y : draw_y + size, draw_x : draw_x + size] = miniframes[cur_id]


def read_individual_miniframes(
    frame: np.ndarray, ordered_centroid: np.ndarray, miniframes: np.ndarray
):
    if frame.ndim == 2:
        frame = frame[..., None]
    miniframes[:] = 0
    size2 = miniframes.shape[1] // 2
    for cur_id, (x, y) in enumerate(ordered_centroid):
        if x > 0 and y > 0:
            miniframe = frame[
                max(0, y - size2) : y + size2, max(0, x - size2) : x + size2
            ]
            miniframes[cur_id, 0 : miniframe.shape[0], 0 : miniframe.shape[1]] = (
                miniframe
            )


def generate_individual_video(
    video: Video,
    trajectories: np.ndarray,
    draw_in_gray: bool,
    starting_frame: int,
    ending_frame: int | None,
):
    if draw_in_gray:
        logging.info("Drawing original video in grayscale")

    trajectories = np.nan_to_num(trajectories, nan=-1).astype(int)

    create_dir(video.individual_videos_folder)

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

    videoPathHolder = VideoPathHolder(video.video_paths)

    ending_frame = len(trajectories) - 1 if ending_frame is None else ending_frame
    logging.info(f"Drawing from frame {starting_frame} to {ending_frame}")

    general_video_writer = cv2.VideoWriter(
        str(video.individual_videos_folder / "general.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        video.frames_per_second,
        (out_video_width, out_video_height),
    )

    individual_video_writers = [
        cv2.VideoWriter(
            str(video.individual_videos_folder / f"individual_{id+1}.avi"),
            cv2.VideoWriter_fourcc(*"XVID"),
            video.frames_per_second,
            (miniframe_size, miniframe_size),
        )
        for id in range(video.number_of_animals)
    ]

    labels = video.identities_labels or list(
        map(str, range(1, video.number_of_animals + 1))
    )

    miniframes = np.empty(
        (video.number_of_animals, miniframe_size, miniframe_size, 3), np.uint8
    )

    general_frame = np.zeros((out_video_height, out_video_width, 3), np.uint8)
    for cur_id in range(video.number_of_animals):
        draw_x, draw_y = positions[cur_id]
        general_frame = cv2.putText(
            general_frame,
            labels[cur_id],
            (draw_x, draw_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )

    for frame in track(range(starting_frame, ending_frame), "Generating video"):
        img = videoPathHolder.read_frame(frame, not draw_in_gray)

        read_individual_miniframes(img, trajectories[frame], miniframes)

        draw_general_frame(positions, miniframe_size, miniframes, general_frame)

        general_video_writer.write(general_frame)

        for id in range(video.number_of_animals):
            individual_video_writers[id].write(miniframes[id])

    logging.info(f"Videos generated in {video.individual_videos_folder}")
