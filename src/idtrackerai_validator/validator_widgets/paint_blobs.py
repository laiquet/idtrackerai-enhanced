from typing import Iterable, Sequence

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QColorConstants, QImage, QPainter, QPolygon

from idtrackerai import Blob
from idtrackerai_GUI_tools import CanvasPainter


def find_selected_blob(
    blobs_in_frame: list[Blob],
    selected_id: int | None,
    last_position: tuple[float, float] | None,
) -> tuple[Blob | None, tuple[float, float] | None]:
    selected_blobs: list[tuple[Blob, tuple[float, float]]] = []
    for blob in blobs_in_frame:
        for identity, centroid in blob.all_final_ids_and_centroids:
            if identity == selected_id:
                selected_blobs.append((blob, centroid))

    if len(selected_blobs) == 0:
        return None, last_position
    if len(selected_blobs) == 1:
        return selected_blobs[0]

    if last_position is None:
        return selected_blobs[0]
    prev_cx, prev_cy = last_position
    return min(
        selected_blobs,
        key=lambda blob: (blob[1][0] - prev_cx) ** 2 + (blob[1][1] - prev_cy) ** 2,
    )


def paintBlobs(
    draw_contours: bool,
    draw_centroids: bool,
    draw_bboxes: bool,
    draw_labels: bool,
    painter: CanvasPainter,
    blobs_in_frame: list[Blob],
    cmap: Sequence[QColor],
    cmap_alpha: Sequence[QColor],
    selected_blob: Blob | None,
    selected_centroid: tuple[float, float] | None,
    labels: list[str],
    marked_blobs: Iterable[Blob],
):
    labels_to_draw: list[tuple[QColor, str, tuple]] = []
    polygon = QPolygon()

    if selected_blob is not None:
        selected_blob_final_identities = list(selected_blob.final_identities)
        color_indx = (
            selected_blob_final_identities[0]
            if len(selected_blob_final_identities) == 1
            and selected_blob_final_identities[0] is not None
            else 0
        )
        color_alpha = cmap_alpha[color_indx]

        painter.setPenColor(QColorConstants.White)
        polygon.setPoints(*selected_blob.contour.ravel())
        painter.setBrush(color_alpha)
        painter.drawPolygon(polygon)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 0, 0, 128))
    for blob in marked_blobs:
        polygon.setPoints(*blob.contour.ravel())
        painter.drawPolygon(polygon)

    for blob in blobs_in_frame:
        blob_final_identities = list(blob.final_identities)
        color_indx = (
            blob_final_identities[0]
            if len(blob_final_identities) == 1 and blob_final_identities[0] is not None
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

        for identity, centroid in blob.final_ids_and_centroids:
            if identity in (None, 0):
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
        painter.setPenColor(QColorConstants.Black)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(x - radius / 2, y - radius / 2, radius, radius))

    # colored centroids
    if draw_centroids:
        painter.setPen(Qt.PenStyle.NoPen)
        for color, _idstr, (x, y) in labels_to_draw:
            painter.setBrush(color)
            painter.drawBigPoint(x, y)

    # labels lines
    if draw_labels:
        zoom = painter.applied_zoom
        for color, idstr, (x, y) in labels_to_draw:
            if idstr:
                pointA = QPointF(x + 25 * zoom, y - 25 * zoom)
                pointB = QPointF(x, y)
                color.setAlpha(90)
                painter.setPenColor(color)
                painter.drawLine(pointA, pointB)

    # black centroid contour
    if draw_centroids:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPenColor(QColorConstants.Black)
        for _color, _idstr, (x, y) in labels_to_draw:
            painter.drawBigPoint(x, y)

    # label text
    if draw_labels:
        zoom = painter.applied_zoom
        for color, idstr, (x, y) in labels_to_draw:
            if idstr:
                pointA = QPointF(x + 25 * zoom, y - 25 * zoom)
                color.setAlpha(255)
                painter.setPenColor(color)
                painter.drawText(pointA, idstr)


def paintTrails(
    frame_number: int,
    painter: CanvasPainter,
    trajectories: np.ndarray,
    cmap: Sequence[QColor],
):
    trail_length = 30
    trail_origin = None if frame_number < trail_length else frame_number - trail_length
    canvas = QImage(
        painter.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied
    )
    canvas.fill(Qt.GlobalColor.transparent)
    trail_painter = QPainter(canvas)
    trail_painter.setWindow(painter.window())

    trail_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    trail_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)

    pen = trail_painter.pen()
    pen.setWidthF(2 * painter.applied_zoom)
    alphas = np.linspace(255, 0, trail_length, False, dtype=np.uint8)
    for cur_id in range(trajectories.shape[1]):
        centroids_trace = trajectories[frame_number:trail_origin:-1, cur_id]
        color = QColor(cmap[cur_id + 1])
        for alpha, pointA, pointB in zip(
            alphas, centroids_trace[1:], centroids_trace[:-1]
        ):
            color.setAlpha(alpha)
            pen.setColor(color)
            trail_painter.setPen(pen)
            trail_painter.drawLine(QPointF(*pointA), QPointF(*pointB))
    trail_painter.end()
    painter.drawImage(painter.window(), canvas)
