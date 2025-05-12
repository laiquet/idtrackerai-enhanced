import logging
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPainter

from idtrackerai import Session
from idtrackerai.GUI_tools import VideoPathHolder, get_cmap
from idtrackerai.utils import load_trajectories, track


def _QImageToArray(qimg: QImage) -> np.ndarray:
    width = qimg.width()
    height = qimg.height()
    byte_str = qimg.bits()
    assert byte_str is not None
    return np.frombuffer(byte_str.asstring(height * width * 4), np.uint8).reshape(
        (height, width, 4)
    )[:, :, :-1]


def _draw_general_frame(
    np_frame: np.ndarray,
    frame_number: int,
    trajectories: np.ndarray,
    centroid_trace_length: int,
    colors: list[tuple[int, int, int]] | np.ndarray,
    labels: list[str],
) -> np.ndarray:
    ordered_centroid = trajectories[frame_number]
    match np_frame.ndim:
        case 3:
            frame = QImage(
                np_frame.data,
                np_frame.shape[1],
                np_frame.shape[0],
                np_frame.shape[1] * 3,
                QImage.Format.Format_BGR888,
            )
        case 2:
            frame = QImage(
                np_frame.data,
                np_frame.shape[1],
                np_frame.shape[0],
                np_frame.shape[1],
                QImage.Format.Format_Grayscale8,
            )
        case _:
            raise RuntimeError("Invalid frame shape ", np_frame.shape)
    canvas = QImage(frame.size(), QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    pen = painter.pen()
    pen.setWidth(2)
    for cur_id, centroid in enumerate(ordered_centroid):
        if frame_number > centroid_trace_length:
            centroids_trace = trajectories[
                frame_number - centroid_trace_length : frame_number + 1, cur_id
            ]
        else:
            centroids_trace = trajectories[: frame_number + 1, cur_id]
        color = QColor(*colors[cur_id])

        alphas = np.linspace(0, 255, len(centroids_trace), dtype=int)[1:]
        if len(centroids_trace) > 1:
            for alpha, (pointA, pointB) in zip(alphas, pairwise(centroids_trace)):
                if any(pointA < 0) or any(pointB < 0):
                    continue
                color.setAlpha(alpha)
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawLine(*pointA, *pointB)

        if all(centroid > 0):
            color.setAlpha(255)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(centroid[0] - 3, centroid[1] - 3, 6, 6)

    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
    painter.drawImage(canvas.rect(), frame)
    painter.end()

    arr_img = np.array(_QImageToArray(canvas))
    if not labels:
        return arr_img

    for cur_id, centroid in enumerate(ordered_centroid):
        if all(centroid > 0):
            color = (
                int(colors[cur_id][2]),
                int(colors[cur_id][1]),
                int(colors[cur_id][0]),
            )  # BGR

            arr_img = cv2.putText(
                arr_img,
                labels[cur_id],
                (centroid[0], centroid[1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color,
                thickness=2,
            )

    return arr_img


def generate_trajectories_video(
    session: Session,
    trajectories_path: Path | str | None = None,
    draw_in_gray: bool = False,
    centroid_trace_length: int = 20,
    starting_frame: int = 0,
    ending_frame: int | None = None,
    no_labels: bool = False,
) -> None:
    """Generate general video, called by the command ``idtrackerai_video``.

    .. seealso::
        Documentation for :ref:`video generators`


    Parameters
    ----------
    session : Session
        _description_
    trajectories_path : Path | str | None, optional
        Path to the trajectories file. If None, the trajectories are loaded from the session folder, by default None.
    draw_in_gray : bool, optional
        Flag to draw the video in grayscale, by default False.
    centroid_trace_length : int, optional
        _description_, by default 20
    starting_frame : int, optional
        Starting frame for the generated video, by default 0.
    ending_frame : int | None, optional
        Ending frame for the generated video. If None, the video is generated until the end, by default None.
    no_labels : bool, optional
        Flag to hide labels in the video, by default False.
    """
    if draw_in_gray:
        logging.info("Drawing original video in grayscale")

    resize_factor = min(1920 / session.width, 1080 / session.height, 1)

    if resize_factor != 1:
        logging.info(f"Applying resize of factor {resize_factor}")

    trajectories = load_trajectories(trajectories_path or session.trajectories_folder)[
        "trajectories"
    ]
    trajectories = np.nan_to_num(trajectories * resize_factor, nan=-1).astype(int)

    video_name = session.video_paths[0].stem + "_tracked.avi"

    colors = get_cmap(session.n_animals)

    if no_labels:
        labels = []
    else:
        labels = session.identities_labels or list(
            map(str, range(1, session.n_animals + 1))
        )

    path_to_save_video = session.session_folder / video_name

    out_video_width = int(session.width * resize_factor + 0.5)
    out_video_height = int(session.height * resize_factor + 0.5)

    video_writer = cv2.VideoWriter(
        str(path_to_save_video),
        cv2.VideoWriter.fourcc(*"XVID"),
        session.frames_per_second,
        (out_video_width, out_video_height),
    )

    videoPathHolder = VideoPathHolder(session.video_paths)

    ending_frame = len(trajectories) - 1 if ending_frame is None else ending_frame
    logging.info(f"Drawing from frame {starting_frame} to {ending_frame}")

    for frame in track(range(starting_frame, ending_frame), "Generating video"):
        try:
            img = videoPathHolder.read_frame(frame, not draw_in_gray)
            if resize_factor != 1:
                img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        except RuntimeError as exc:
            logging.error(str(exc))
            img = np.zeros(
                (
                    (out_video_height, out_video_width)
                    if draw_in_gray
                    else (out_video_height, out_video_width, 3)
                ),
                np.uint8,
            )

        img = _draw_general_frame(
            img, frame, trajectories, centroid_trace_length, colors, labels
        )

        video_writer.write(img)

    logging.info(f"Video generated in {path_to_save_video}")
