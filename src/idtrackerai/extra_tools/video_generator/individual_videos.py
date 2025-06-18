import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from idtrackerai import Session
from idtrackerai.GUI_tools import VideoPathHolder
from idtrackerai.utils import create_dir, load_trajectories, track


def draw_general_frame(
    positions: list[tuple[int, int]],
    miniframes: np.ndarray,
    output_shape: tuple[int, int],
    labels: list[str],
) -> np.ndarray:
    canvas = np.zeros((*output_shape, 3), np.uint8)
    miniframe_size = miniframes.shape[1]
    for cur_id in range(len(miniframes)):
        draw_x, draw_y = positions[cur_id]
        canvas[draw_y : draw_y + miniframe_size, draw_x : draw_x + miniframe_size] = (
            miniframes[cur_id]
        )
        canvas = cv2.putText(
            canvas,
            labels[cur_id],
            (draw_x, draw_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
    return canvas


def read_individual_miniframes(
    frame: np.ndarray, ordered_centroid: np.ndarray, size: int
) -> np.ndarray:
    if frame.ndim == 2:
        frame = frame[..., None]
    miniframes = np.zeros((len(ordered_centroid), size, size, 3), np.uint8)
    size2 = size // 2
    for cur_id, (x, y) in enumerate(ordered_centroid):
        if x > 0 and y > 0:
            miniframe = frame[
                max(0, y - size2) : y + size2, max(0, x - size2) : x + size2
            ]

            # with the next slices, the centroid is always in the center of the video,
            # even if the miniframe is cropped
            y_location = (
                slice(None, miniframe.shape[0])
                if y > size2
                else slice(-miniframe.shape[0], None)
            )
            x_location = (
                slice(None, miniframe.shape[1])
                if x > size2
                else slice(-miniframe.shape[1], None)
            )
            miniframes[cur_id, y_location, x_location] = miniframe
    return miniframes


def get_geometries(
    n_animals: int, miniframe_size: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n_rows = int(np.sqrt(n_animals))
    n_cols = int(np.ceil(n_animals / n_rows))
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
        for i in range(n_animals)
    ]
    return positions, (out_video_height, out_video_width)


def generate_individual_video(
    session: Session,
    trajectories_path: Path | str | None = None,
    draw_in_gray: bool = False,
    starting_frame: int = 0,
    ending_frame: int | None = None,
    miniframe_size: float | None = None,
    labels: list[str] | None = None,
    callback: Callable | None = None,
) -> None:
    """Generate individual video, called by the command ``idtrackerai_video``.

    .. seealso::
        Documentation for :ref:`video generators`

    Parameters
    ----------
    session : Session
        Session instance to generate the videos from.
    trajectories_path : Path | str | None, optional
        Path to the trajectories file. If None, the trajectories are loaded from the session folder, by default None.
    draw_in_gray : bool, optional
        Flag to draw the video in grayscale, by default False.
    starting_frame : int, optional
        Starting frame for the generated video, by default 0.
    ending_frame : int | None, optional
        Ending frame for the generated video. If None, the video is generated until the end, by default None.
    miniframe_size : float | None, optional
        Size, in pixels, of the individual square videos. If None, the size is adapted to fit the median body length, by default None.
    """
    if draw_in_gray:
        logging.info("Drawing original video in grayscale")

    trajectories = load_trajectories(trajectories_path or session.trajectories_folder)[
        "trajectories"
    ]

    trajectories = np.nan_to_num(trajectories, nan=-1).astype(int)

    create_dir(session.individual_videos_folder)

    miniframe_size = 2 * (int(miniframe_size or session.median_body_length) // 2)
    logging.info(f"Square individual videos set to {miniframe_size} pixels")
    positions, output_shape = get_geometries(session.n_animals, miniframe_size)

    videoPathHolder = VideoPathHolder(session.video_paths)

    ending_frame = len(trajectories) - 1 if ending_frame is None else ending_frame
    logging.info(f"Drawing from frame {starting_frame} to {ending_frame}")

    general_video_writer = cv2.VideoWriter(
        str(session.individual_videos_folder / "general.avi"),
        cv2.VideoWriter.fourcc(*"XVID"),
        session.frames_per_second,
        output_shape[::-1],
    )

    individual_video_writers = [
        cv2.VideoWriter(
            str(session.individual_videos_folder / f"individual_{id + 1}.avi"),
            cv2.VideoWriter.fourcc(*"XVID"),
            session.frames_per_second,
            (miniframe_size, miniframe_size),
        )
        for id in range(session.n_animals)
    ]

    for frame in track(
        range(starting_frame, ending_frame + 1), "Generating video", callback=callback
    ):
        try:
            img = videoPathHolder.read_frame(frame, not draw_in_gray)
        except RuntimeError as exc:
            logging.error(str(exc))
            img = np.zeros(
                (
                    (session.height, session.width)
                    if draw_in_gray
                    else (session.height, session.width, 3)
                ),
                np.uint8,
            )

        miniframes = read_individual_miniframes(
            img, trajectories[frame], miniframe_size
        )

        general_frame = draw_general_frame(positions, miniframes, output_shape, labels)

        general_video_writer.write(general_frame)

        for id in range(session.n_animals):
            individual_video_writers[id].write(miniframes[id])

    logging.info(f"Videos generated in {session.individual_videos_folder}")
