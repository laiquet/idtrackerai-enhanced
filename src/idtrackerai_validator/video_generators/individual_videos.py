import logging
import sys
import cv2
import numpy as np
from idtrackerai_app.widgets_utils import VideoPathHolder
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow
from rich.progress import Progress

from idtrackerai import Video


def QImageToArray(qimg: QImage) -> np.ndarray:
    width = qimg.width()
    height = qimg.height()
    byte_str = qimg.bits()
    byte_str.setsize(height * width * 4)
    return np.frombuffer(byte_str, np.uint8).reshape((height, width, 4))[:, :, :-1]


def draw_individual_frame(
    frame: QImage,
    ordered_centroid: np.ndarray,
    positions: list[tuple[int, int]],
    width,
    height,
    size: int,
) -> QImage:
    canvas = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(Qt.GlobalColor.black)
    painter = QPainter(canvas)
    painter.setPen(Qt.GlobalColor.white)

    size2 = size // 2
    for cur_id, (x, y) in enumerate(ordered_centroid):
        draw_x, draw_y = positions[cur_id]

        painter.drawText(
            draw_x,
            draw_y - 20,
            size,
            19,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            str(cur_id + 1),
        )
        if x > 0 and y > 0:
            painter.drawImage(
                draw_x, draw_y, frame.copy(x - size2, y - size2, size, size)
            )
    return canvas


class GeneralVideoGenerator(QMainWindow):
    def __init__(
        self,
        video: Video,
        trajectories,
        draw_in_gray: bool,
        starting_frame,
        ending_frame,
    ):
        super().__init__()
        self.draw_in_gray = draw_in_gray
        if self.draw_in_gray:
            logging.info(f"Drawing original video in grayscale")

        self.trajectories = trajectories.astype(int)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)

        self.path_to_save_video = video.session_folder / (
            video.video_paths[0].stem + "_individuals.avi"
        )

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        n_rows = int(np.sqrt(video.number_of_animals))
        n_cols = int(video.number_of_animals / n_rows - 0.0001) + 1

        self.miniframe_size = int(video.median_body_length_full_resolution)
        extra_lower_pad = 10
        bbox_side_pad = 10
        bbox_top_pad = 30
        full_bbox_width = self.miniframe_size + 2 * bbox_side_pad
        self.out_video_width = n_cols * full_bbox_width

        full_bbox_height = self.miniframe_size + bbox_top_pad
        self.out_video_height = n_rows * full_bbox_height + extra_lower_pad

        self.setFixedSize(self.out_video_width, self.out_video_height)
        self.positions = [
            (
                full_bbox_width * (i % n_cols) + bbox_side_pad,
                full_bbox_height * (i // n_cols) + bbox_top_pad,
            )
            for i in range(video.number_of_animals)
        ]

        self.video_writer = cv2.VideoWriter(
            str(self.path_to_save_video),
            fourcc,
            video.frames_per_second,
            (self.out_video_width, self.out_video_height),
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

        if self.draw_in_gray:
            img = cv2.cvtColor(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB
            )

        Qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        img = draw_individual_frame(
            Qimg,
            self.trajectories[self.current_frame],
            self.positions,
            width=self.out_video_width,
            height=self.out_video_height,
            size=self.miniframe_size,
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


def generate_individual_video(
    video, trajectories, draw_in_gray, starting_frame, ending_frame, **kargs
):
    app = QApplication(sys.argv)
    window = GeneralVideoGenerator(
        video, trajectories, draw_in_gray, starting_frame, ending_frame
    )
    window.show()
    app.exec()
