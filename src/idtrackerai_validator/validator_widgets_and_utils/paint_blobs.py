import numpy as np
from idtrackerai_app.widgets_utils import CustomQPainter
from PyQt6.QtCore import QPointF, Qt, QRectF
from PyQt6.QtGui import QColor, QPolygon

from idtrackerai import Blob


def find_selected_blob(
    blobs_in_frame: list[Blob],
    selected_id: int | None,
    last_position: tuple[float, float] | None,
) -> tuple[Blob | None, tuple[float, float] | None]:

    if selected_id is None:
        return None, None
    selected_blobs: list[tuple[Blob, tuple[float, float]]] = []
    for blob in blobs_in_frame:
        for identity, centroid in zip(blob.final_identities, blob.final_centroids):
            if identity not in (None, -1) and identity == selected_id:
                selected_blobs.append((blob, centroid))

    if len(selected_blobs) == 0:
        return None, last_position
    elif len(selected_blobs) == 1:
        return selected_blobs[0]
    else:
        if last_position is not None:
            prev_cx, prev_cy = last_position
            return sorted(
                selected_blobs,
                key=lambda blob: (blob[1][0] - prev_cx) ** 2
                + (blob[1][1] - prev_cy) ** 2,
            )[0]
        else:
            return selected_blobs[0]


def paintBlobs(
    draw_contours: bool,
    draw_centroids: bool,
    draw_bboxes: bool,
    draw_labels: bool,
    painter: CustomQPainter,
    blobs_in_frame: list[Blob],
    segments: np.ndarray,
    cmap: list[QColor],
    cmap_alpha: list[QColor],
    selected_blob: Blob | None,
    selected_centroid: tuple[float, float] | None,
    labels: list[str],
):
    labels_to_draw = []
    polygon = QPolygon()
    # trail_origin = max(0, frame_number - self.trail_length)

    if selected_blob is not None:

        color_indx = (
            selected_blob.final_identities[0]
            if len(selected_blob.final_identities) == 1
            and selected_blob.final_identities[0] is not None
            else 0
        )
        color_alpha = cmap_alpha[color_indx]

        painter.setPenColor(QColor("white"))
        polygon.setPoints(*selected_blob.contour.ravel())
        painter.setBrush(color_alpha)
        painter.drawPolygon(polygon)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    for blob in blobs_in_frame:
        color_indx = (
            blob.final_identities[0]
            if len(blob.final_identities) == 1 and blob.final_identities[0] is not None
            else 0
        )
        color = cmap[color_indx]

        painter.setPenColor(color)

        if draw_contours:
            polygon.setPoints(*blob.contour.ravel())
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(polygon)

        if draw_bboxes:
            (x0, y0), (x1, y1) = blob.bbox_in_frame_coordinates
            polygon.setPoints(x0, y0, x1, y0, x1, y1, x0, y1)
            painter.drawPolygon(polygon)

        for identity, centroid in zip(blob.final_identities, blob.final_centroids):
            if identity in (None, -1, 0):
                idstr = ""
            else:
                if (
                    blob.user_generated_identities is not None
                    and identity in blob.user_generated_identities
                    and blob.user_generated_centroids is not None
                    and centroid in blob.user_generated_centroids
                ):
                    idstr = "u-" + labels[identity]

                elif (
                    blob.identities_corrected_closing_gaps is not None
                    and not blob.is_an_individual
                ):
                    idstr = "c-" + labels[identity]
                else:
                    idstr = labels[identity]

            color = cmap[0 if identity is None else identity]
            labels_to_draw.append((color, idstr, centroid))

    if selected_blob is not None and selected_centroid is not None:
        radius = 15 * painter.applied_zoom
        x, y = selected_centroid
        painter.setPenColor(QColor("black"))
        painter.drawEllipse(QRectF(x - radius / 2, y - radius / 2, radius, radius))

    # colored centroids
    if draw_centroids:
        painter.setPen(Qt.PenStyle.NoPen)
        for color, idstr, (x, y) in labels_to_draw:
            painter.setBrush(color)
            painter.drawBigPoint(x, y)

    # labels lines
    if draw_labels:
        zoom = painter.applied_zoom
        for color, idstr, (x, y) in labels_to_draw:
            if idstr:
                pointA = QPointF(x + 25 * zoom, y - 25 * zoom)
                pointB = QPointF(x, y)
                painter.setPenColor(color)
                painter.drawLine(pointA, pointB)

    # black centroid contour
    if draw_centroids:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPenColor(QColor("black"))
        for color, idstr, (x, y) in labels_to_draw:
            painter.drawBigPoint(x, y)

    # label text
    if draw_labels:
        zoom = painter.applied_zoom
        for color, idstr, (x, y) in labels_to_draw:
            if idstr:
                pointA = QPointF(x + 25 * zoom, y - 25 * zoom)
                painter.setPenColor(color)
                painter.drawText(pointA, idstr)

    # for id, trail in enumerate(self.trails):
    #     trail.set_segments(segments[trail_origin:frame_number, id])
