import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from idtrackerai_app.widgets_utils import VideoPathHolder
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from rich.progress import Progress
from idtrackerai import Video


def QImageToArray(qimg: QImage) -> np.ndarray:
    width = qimg.width()
    height = qimg.height()
    byte_str = qimg.bits()
    byte_str.setsize(height * width * 4)
    return np.frombuffer(byte_str, np.uint8).reshape((height, width, 4))[:, :, :-1]


def writeIds(
    frame: QImage,
    frame_number: int,
    trajectories: np.ndarray,
    centroid_trace_length: int,
    colors: list[tuple[int, int, int]],
):
    ordered_centroid = trajectories[frame_number]

    canvas = QImage(frame.size(), QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    font = painter.font()
    pen = painter.pen()
    font.setPointSize(25)
    painter.setFont(font)
    for cur_id, centroid in enumerate(ordered_centroid):
        if frame_number > centroid_trace_length:
            centroids_trace = trajectories[
                frame_number - centroid_trace_length : frame_number + 1, cur_id
            ]
        else:
            centroids_trace = trajectories[: frame_number + 1, cur_id]
        color = QColor(*colors[cur_id])
        int_centroid = np.asarray(centroid, int)

        pen.setWidth(3)
        if len(centroids_trace) > 1:
            centroids_trace = centroids_trace.astype(int)
            alphas = np.linspace(0, 255, len(centroids_trace), dtype=int)[1:]

            for alpha, pointA, pointB in zip(
                alphas, centroids_trace[1:], centroids_trace[:-1]
            ):
                if any(pointA < 0) or any(pointB < 0):
                    continue
                color.setAlpha(alpha)
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawLine(*pointA, *pointB)

        if not any(np.isnan(centroid)):
            color.setAlpha(255)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int_centroid[0] - 3, int_centroid[1] - 3, 6, 6)
            painter.setPen(color)
            painter.drawText(int_centroid[0], int_centroid[1], str(cur_id + 1))

    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
    painter.drawImage(canvas.rect(), frame)

    return canvas


class GeneralVideoGenerator(QMainWindow):
    def __init__(
        self,
        video: Video,
        trajectories,
        draw_in_gray: bool,
        centroid_trace_length,
        starting_frame,
        ending_frame,
    ):
        super().__init__()
        self.draw_in_gray = draw_in_gray
        if self.draw_in_gray:
            logging.info(f"Drawing original video in grayscale")

        self.resize_factor = min(
            1920 / video.original_width, 1080 / video.original_height, 1
        )

        if self.resize_factor != 1:
            logging.info(f"Applying resize of factor {self.resize_factor}")

        self.trajectories = trajectories * self.resize_factor
        self.centroid_trace_length = centroid_trace_length
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)

        video_name = (
            os.path.split(video.video_paths[0])[-1].split(".")[0] + "_tracked.avi"
        )

        parent_dir = Path(__file__).parent.parent
        for file in parent_dir.glob("cmap_*"):
            general_cmap = np.loadtxt(parent_dir / file, dtype=int)

        self.colors = [
            general_cmap[int(i * 255 / video.number_of_animals)]
            for i in range(video.number_of_animals)
        ]

        self.path_to_save_video = video.session_folder / video_name
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        out_video_width = int(video.original_width * self.resize_factor)
        out_video_height = int(video.original_height * self.resize_factor)

        self.setFixedSize(int(0.8 * out_video_width), int(0.8 * out_video_height))

        self.video_writer = cv2.VideoWriter(
            str(self.path_to_save_video),
            fourcc,
            video.frames_per_second,
            (out_video_width, out_video_height),
        )

        self.videoPathHolder = VideoPathHolder(video.video_paths)
        timer = QTimer(self)
        timer.timeout.connect(self.process_frame)
        timer.start()
        self.current_frame = starting_frame

        self.ending_frame = (
            len(trajectories) - 1 if ending_frame is None else ending_frame
        )
        logging.info(f"Drawing from frame {self.current_frame} to {self.ending_frame}")

        self.progress = Progress()
        self.progress.start()
        self.task1 = self.progress.add_task(
            "[red]Rendering video", total=self.ending_frame - self.current_frame
        )
        QTimer.singleShot(0, self.center_window)

    def process_frame(self):

        img = self.videoPathHolder.frameColor(self.current_frame)

        if self.resize_factor != 1:
            img = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

        if self.draw_in_gray:
            img = cv2.cvtColor(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB
            )

        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        img = writeIds(
            img,
            self.current_frame,
            self.trajectories,
            self.centroid_trace_length,
            self.colors,
        )

        self.label.setPixmap(QPixmap.fromImage(img))

        self.video_writer.write(QImageToArray(img))
        self.current_frame += 1
        self.progress.update(self.task1, advance=1)

        if self.current_frame == self.ending_frame:
            self.progress.stop()
            logging.info(f"Video generated in {self.path_to_save_video}")
            self.close()

    def center_window(self):
        w = self.width()
        h = self.height()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        self.setGeometry(cp.x() - w // 2, max(5, cp.y() - h) // 2, w, h)


def generate_trajectories_video(
    video,
    trajectories,
    draw_in_gray,
    centroid_trace_length,
    starting_frame,
    ending_frame,
):
    app = QApplication(sys.argv)
    window = GeneralVideoGenerator(
        video,
        trajectories,
        draw_in_gray,
        centroid_trace_length,
        starting_frame,
        ending_frame,
    )
    window.show()
    app.exec()
    logging
