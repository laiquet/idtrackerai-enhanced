import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.patches import Polygon, Rectangle

from idtrackerai import Blob


class BlobsArtists:
    def __init__(self, n_blobs: int, ax: Axes, cmap: np.ndarray):

        self.cmap = cmap
        self.cmap_alpha = np.column_stack((cmap, np.full(len(cmap), 0.3)))
        self.contours = [ax.add_patch(Polygon([[0, 0]], facecolor="None", picker=True)) for _ in range(n_blobs)]  # type: ignore
        self.bboxes = [
            ax.add_patch(Rectangle((0, 0), 0.0, 0.0, facecolor="None"))
            for _ in range(n_blobs)
        ]
        self.centroids = ax.scatter(x=[], y=[], s=5.0)
        self.labels = [
            ax.annotate(
                "",
                [0, 0],
                xytext=[-30.0, 30.0],
                textcoords="offset pixels",
                verticalalignment="center",
                horizontalalignment="right",
                fontsize="x-large",
                arrowprops={"arrowstyle": "-"},
            )
            for _ in range(n_blobs)
        ]

    def set_blobs(self, blobs: list[Blob], selected_fragment: int = -1) -> Blob | None:
        selected_blob = None
        centroids_colors = []
        centroids_positions = []
        centroid_indx = 0
        for contour, bbox in zip(self.contours, self.bboxes):
            contour.set_visible(False)
            bbox.set_visible(False)

        for blob, contour, bbox in zip(blobs, self.contours, self.bboxes):
            color_indx = (
                blob.final_identities[0]
                if len(blob.final_identities) == 1
                and blob.final_identities[0] is not None
                else 0
            )
            color = self.cmap[color_indx]
            color_alpha = self.cmap_alpha[color_indx]
            if blob.fragment_identifier == selected_fragment:
                selected_blob = blob

            contour.set(
                xy=blob.contour,
                edgecolor=color,
                visible=True,
                facecolor=color_alpha
                if blob.fragment_identifier == selected_fragment
                else "None",
            )
            contour.associated_blob = blob
            ((x0, y0), (x1, y1)) = blob.bbox_in_frame_coordinates
            bbox.set(bounds=(x0, y0, x1 - x0, y1 - y0), edgecolor=color, visible=True)

            centroids_colors.extend(
                self.cmap[0 if identity is None else identity]
                for identity in blob.final_identities
            )
            centroids_positions.extend(blob.final_centroids)

            for identity, centroid in zip(
                blob.final_identities, blob.final_centroids_full_resolution
            ):
                color = self.cmap[0 if identity is None else identity]
                self.labels[centroid_indx].set(
                    color=color, text=str(identity), visible=True
                )
                self.labels[centroid_indx].xy = centroid
                self.labels[centroid_indx].arrow_patch.set(edgecolor=color)
                centroid_indx += 1

        for annotation in self.labels[centroid_indx:]:
            annotation.set_visible(False)
        self.centroids.set(offsets=centroids_positions, facecolor=centroids_colors)

        return selected_blob

    def draw_contours(self, renderer: RendererAgg):
        for contour in self.contours:
            contour.draw(renderer)

    def draw_bboxes(self, renderer: RendererAgg):
        for bbox in self.bboxes:
            bbox.draw(renderer)

    def draw_centroids(self, renderer: RendererAgg):
        self.centroids.draw(renderer)

    def draw_labels(self, renderer: RendererAgg):
        for label in self.labels:
            label.draw(renderer)
