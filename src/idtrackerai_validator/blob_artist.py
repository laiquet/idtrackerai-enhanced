import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle

from idtrackerai import Blob


class BlobsArtists:
    def __init__(self, n_blobs: int, ax: Axes, cmap: np.ndarray):
        self.n_animals = n_blobs
        self.ax = ax
        self.cmap = cmap
        self.cmap_alpha = np.column_stack((cmap, np.full(len(cmap), 0.3)))
        self.contours = [ax.add_patch(Polygon([[0, 0]], facecolor="None")) for _ in range(n_blobs)]  # type: ignore
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
        self.trail_length = 30
        self.trails: list[LineCollection] = []  # DON'T ASK...
        for i in range(n_blobs):
            color = np.tile(self.cmap_alpha[i + 1], (self.trail_length, 1))
            color[:, -1] = np.linspace(0, 1, self.trail_length)
            self.trails.append(ax.add_collection(LineCollection([], color=color)))  # type: ignore

    def set_blobs(
        self,
        blobs_in_video: list[list[Blob]],
        frame_number: int,
        segments: np.ndarray,
        selected_fragment: int = -1,
    ) -> Blob | None:
        selected_blob = None
        centroids_colors = []
        centroids_positions = []
        centroid_indx = 0
        trail_origin = max(0, frame_number - self.trail_length)
        for contour, bbox in zip(self.contours, self.bboxes):
            contour.set_visible(False)
            bbox.set_visible(False)

        for blob, contour, bbox in zip(
            blobs_in_video[frame_number], self.contours, self.bboxes
        ):
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

                try:
                    self.labels[centroid_indx].set(
                        color=color, text=idstr, visible=True
                    )
                    self.labels[centroid_indx].xy = centroid
                    self.labels[centroid_indx].arrow_patch.set(edgecolor=color)
                except IndexError:
                    self.labels.append(
                        self.ax.annotate(
                            idstr,
                            centroid,
                            xytext=[-30.0, 30.0],
                            textcoords="offset pixels",
                            verticalalignment="center",
                            horizontalalignment="right",
                            fontsize="x-large",
                            arrowprops={"arrowstyle": "-", "edgecolor": color},
                            color=color,
                        )
                    )

                centroid_indx += 1

        for id, trail in enumerate(self.trails):
            trail.set_segments(segments[trail_origin:frame_number, id])

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

    def draw_trails(self, renderer: RendererAgg):
        for trail in self.trails:
            trail.draw(renderer)
