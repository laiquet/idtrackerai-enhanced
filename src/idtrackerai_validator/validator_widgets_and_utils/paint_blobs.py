import numpy as np
from idtrackerai_app.widgets_utils import CustomQPainter
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPolygon

from idtrackerai import Blob


def paintBlobs(
    draw_contours: bool,
    draw_centroids: bool,
    draw_bboxes: bool,
    draw_labels: bool,
    painter: CustomQPainter,
    blobs_in_video: list[list[Blob]],
    frame_number: int,
    segments: np.ndarray,
    cmap: list[QColor],
    cmap_alpha: list[QColor],
    selected_fragment: int = -1,
    selected_id: int = -1,
) -> Blob | None:
    selected_blob = None
    labels_to_draw = []
    polygon = QPolygon()
    # trail_origin = max(0, frame_number - self.trail_length)
    pen = painter.pen()

    for blob in blobs_in_video[frame_number]:
        color_indx = (
            blob.final_identities[0]
            if len(blob.final_identities) == 1 and blob.final_identities[0] is not None
            else 0
        )
        color = cmap[color_indx]
        color_alpha = cmap_alpha[color_indx]

        pen.setColor(color)
        painter.setPen(pen)

        if (
            selected_fragment == blob.fragment_identifier
            or selected_id in blob.final_identities
        ):
            selected_blob = blob

        if draw_contours:
            polygon.setPoints(*blob.contour.ravel())
            if selected_blob == blob:
                painter.setBrush(color_alpha)
                painter.drawPolygon(polygon)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(polygon)

        if draw_bboxes:
            (x0, y0), (x1, y1) = blob.bbox_in_frame_coordinates
            polygon.setPoints(x0, y0, x1, y0, x1, y1, x0, y1)
            painter.drawPolygon(polygon)

        for identity, centroid in zip(blob.final_identities, blob.final_centroids):
            if (
                blob.user_generated_identities is not None
                and identity in blob.user_generated_identities
                and blob.user_generated_centroids is not None
                and centroid in blob.user_generated_centroids
            ):
                idstr = f"u-{identity}"
            elif (
                blob.identities_corrected_closing_gaps is not None
                and not blob.is_an_individual
            ):
                idstr = f"c-{identity}"
            else:
                idstr = f"{identity}"

            color = cmap[0 if identity is None else identity]

            if draw_centroids:
                pen.setColor(color)
                painter.setPen(pen)
                painter.setBrush(color)
                painter.drawBigPoint(*centroid)

            labels_to_draw.append((color, idstr, centroid))

    if draw_labels:
        zoom = painter.applied_zoom
        for color, idstr, (x, y) in labels_to_draw:
            pointA = QPointF(x + 25 * zoom, y - 25 * zoom)
            pointB = QPointF(x, y)
            pen.setColor(color)
            painter.setPen(pen)
            painter.drawText(pointA, idstr)
            painter.drawLine(pointA, pointB)

    # for id, trail in enumerate(self.trails):
    #     trail.set_segments(segments[trail_origin:frame_number, id])

    return selected_blob
