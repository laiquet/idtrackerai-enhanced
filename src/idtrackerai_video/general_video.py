import logging

import cv2
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPainter

from idtrackerai import Video
from idtrackerai.utils import track
from idtrackerai_GUI_tools import VideoPathHolder, get_cmap


def QImageToArray(qimg: QImage) -> np.ndarray:
    width = qimg.width()
    height = qimg.height()
    byte_str = qimg.bits()
    return np.frombuffer(byte_str.asarray(height * width * 4), np.uint8).reshape(
        (height, width, 4)
    )[:, :, :-1]


def setColormap(n_animals: int):
    return get_cmap()[[int(i * 255 / n_animals) for i in range(n_animals)]]


def draw_general_frame(
    np_frame: np.ndarray,
    frame_number: int,
    trajectories: np.ndarray,
    centroid_trace_length: int,
    colors: list[tuple[int, int, int]],
    labels: list[str],
) -> np.ndarray:
    ordered_centroid = trajectories[frame_number]
    frame = QImage(
        np_frame.data,
        np_frame.shape[1],
        np_frame.shape[0],
        np_frame.shape[1] * 3,
        QImage.Format.Format_BGR888,
    )
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
            for alpha, pointA, pointB in zip(
                alphas, centroids_trace[1:], centroids_trace[:-1]
            ):
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

    arr_img = np.array(QImageToArray(canvas))
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
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                color,
                2,
            )

    return arr_img


def generate_trajectories_video(
    video: Video,
    trajectories: np.ndarray,
    draw_in_gray: bool,
    centroid_trace_length: int,
    starting_frame: int,
    ending_frame: int,
):
    if draw_in_gray:
        logging.info("Drawing original video in grayscale")

    resize_factor = min(1920 / video.original_width, 1080 / video.original_height, 1)

    if resize_factor != 1:
        logging.info(f"Applying resize of factor {resize_factor}")

    trajectories = np.nan_to_num(trajectories * resize_factor, nan=-1).astype(int)

    video_name = video.video_paths[0].stem + "_tracked.avi"

    colors = setColormap(video.number_of_animals)

    labels = video.identities_labels or list(
        map(str, range(1, video.number_of_animals + 1))
    )

    path_to_save_video = video.session_folder / video_name

    out_video_width = int(video.original_width * resize_factor)
    out_video_height = int(video.original_height * resize_factor)

    video_writer = cv2.VideoWriter(
        str(path_to_save_video),
        cv2.VideoWriter_fourcc(*"XVID"),
        video.frames_per_second,
        (out_video_width, out_video_height),
    )

    videoPathHolder = VideoPathHolder(video.video_paths)

    ending_frame = len(trajectories) - 1 if ending_frame is None else ending_frame
    logging.info(f"Drawing from frame {starting_frame} to {ending_frame}")

    for frame in track(range(starting_frame, ending_frame), "Generating video"):
        img = videoPathHolder.read_frame(frame, True)

        if resize_factor != 1:
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        if draw_in_gray:
            img = cv2.cvtColor(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )

        img = draw_general_frame(
            img, frame, trajectories, centroid_trace_length, colors, labels
        )

        video_writer.write(img)

    logging.info(f"Video generated in {path_to_save_video}")
