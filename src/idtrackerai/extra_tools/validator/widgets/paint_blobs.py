from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np
from qtpy.QtCore import QPointF, QRect, QRectF, Qt
from qtpy.QtGui import QColor, QColorConstants, QImage, QPainter

from idtrackerai import Blob
from idtrackerai.GUI_tools import CanvasPainter


def find_selected_blob(
    blobs_in_frame: list[Blob],
    selected_id: int | None,
    last_position: tuple[float, float] | None,
) -> tuple[Blob | None, tuple[float, float] | None]:
    selected_blobs: list[tuple[Blob, tuple[float, float]]] = []
    for blob in blobs_in_frame:
        for identity, centroid in blob.final_ids_and_centroids:
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
    painter_window = painter.window()

    if selected_blob is not None:
        selected_blob_final_identities = list(selected_blob.final_identities)
        color_indx = (
            selected_blob_final_identities[0]
            if len(selected_blob_final_identities) == 1
            and selected_blob_final_identities[0] is not None
            else 0
        )
        color_alpha = cmap_alpha[color_indx]

        painter.setPen(QColorConstants.White)
        painter.setBrush(color_alpha)
        painter.drawPolygonFromVertices(selected_blob.contour)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 0, 0, 128))
    for blob in marked_blobs:
        painter.drawPolygonFromVertices(blob.contour)

    for blob in blobs_in_frame:
        x0, y0, x1, y1 = blob.bbox_corners
        if not painter_window.intersects(QRect(x0, y0, x1 - x0, y1 - y0)):
            continue

        blob_final_identities = list(blob.final_identities)
        color_indx = (
            blob_final_identities[0]
            if len(blob_final_identities) == 1 and blob_final_identities[0] is not None
            else 0
        )
        color = cmap[color_indx]

        painter.setPen(color)

        if draw_contours:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygonFromVertices(blob.contour)

        if draw_bboxes:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            bbox = blob.bbox_corners
            painter.drawPolygonFromVertices(
                (
                    (bbox.bottom, bbox.left),
                    (bbox.top, bbox.left),
                    (bbox.top, bbox.right),
                    (bbox.bottom, bbox.right),
                )
            )

        for identity, centroid in blob.final_ids_and_centroids:
            if not painter_window.contains(int(centroid[0]), int(centroid[1])):
                continue
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
        painter.setPen(QColorConstants.Black)
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
                color.setAlpha(90)
                painter.setPen(color)
                painter.drawLine(QPointF(x + 25 * zoom, y - 25 * zoom), QPointF(x, y))

    # black centroid contour
    if draw_centroids:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColorConstants.Black)
        for _color, _idstr, (x, y) in labels_to_draw:
            painter.drawBigPoint(x, y)

    # label text
    if draw_labels:
        zoom = painter.applied_zoom
        for color, idstr, (x, y) in labels_to_draw:
            if idstr:
                color.setAlpha(255)
                painter.setPen(color)
                painter.drawText(QPointF(x + 25 * zoom, y - 25 * zoom), idstr)


def paintTrails(
    frame_number: int,
    painter: CanvasPainter,
    trajectories: np.ndarray,
    cmap: Sequence[QColor],
):
    trail_length = 30
    trail_origin = None if frame_number < trail_length else frame_number - trail_length
    trail_points = trajectories[frame_number:trail_origin:-1]

    if np.isnan(trail_points).all():
        return

    min_x, min_y = np.nanmin(trail_points, axis=(0, 1)).astype(int)
    max_x, max_y = np.nanmax(trail_points, axis=(0, 1)).astype(int)
    traces_bbox = QRect(min_x, min_y, max_x - min_x, max_y - min_y)
    drawing_area = painter.window().intersected(traces_bbox)

    if drawing_area.isEmpty():
        return

    canvas = QImage(
        int(drawing_area.width() / painter.applied_zoom),
        int(drawing_area.height() / painter.applied_zoom),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    canvas.fill(Qt.GlobalColor.transparent)
    trail_painter = QPainter(canvas)
    trail_painter.setWindow(drawing_area)

    trail_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    trail_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)

    x0 = drawing_area.x()
    y0 = drawing_area.y()
    x1 = x0 + drawing_area.width()
    y1 = y0 + drawing_area.height()

    pen = trail_painter.pen()
    pen.setWidthF(2 * painter.applied_zoom)
    alphas = np.linspace(255, 0, trail_length, False, dtype=np.uint8)

    are_inside = np.any(
        (trail_points[:, :, 0] > x0)
        * (trail_points[:, :, 0] < x1)
        * (trail_points[:, :, 1] > y0)
        * (trail_points[:, :, 1] < y1),
        0,
    )

    for id_trail_points, is_inside, color in zip(
        trail_points.swapaxes(0, 1), are_inside, cmap[1:]
    ):
        if not is_inside:
            continue

        color = QColor(color)
        for alpha, (A, B) in zip(
            alphas, pairwise(QPointF(*xy) for xy in id_trail_points)
        ):
            color.setAlpha(alpha)
            pen.setColor(color)
            trail_painter.setPen(pen)
            trail_painter.drawLine(A, B)
    trail_painter.end()
    painter.drawImage(drawing_area, canvas)
