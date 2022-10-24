import numpy as np, os, sys
from pathlib import Path
from idtrackerai import Video
import sys


def import_session(project, session_path: Path, progress_event=None):
    """

    :param pythonvideoannotator_models.Project project: Python video annotator project
    :param str project_path: IdtrackerAI project.
    :param func progress_event: Function to update the progress. func(progress_count, max_count=None)
    :return:
    """

    blobs_path = (
        session_path / "preprocessing" / "blobs_collection_no_gaps.npy"
    )
    if not blobs_path.exists():
        blobs_path = session_path / "preprocessing" / "blobs_collection.npy"
    vidobj_path = session_path / "video_object.npy"

    blobs = np.load(blobs_path, allow_pickle=True).item()
    video = Video.load(vidobj_path)

    resolution = video.resolution_reduction

    objs = {}
    paths = {}
    crossings = {}
    fragments = {}
    modifications = {}
    idswitchs = {}

    total_blobs = len(blobs.blobs_in_video)

    for frame_index, blobs_in_frame in enumerate(blobs.blobs_in_video):
        for blob in blobs_in_frame:

            identities = blob.final_identities
            centroids = blob.final_centroids_full_resolution
            fragment = blob.fragment_identifier
            crossing = blob.is_a_crossing
            contour = blob.contour_full_resolution

            for identity, centroid in zip(identities, centroids):

                if identity not in objs:
                    obj = video.create_object()
                    obj.name = str(identity)
                    objs[identity] = obj

                    path = obj.create_path()
                    path.show_object_name = True
                    path.name = "path"
                    paths[identity] = path

                    cnt = obj.create_contours()
                    cnt.name = "contours"

                    c = obj.create_value()
                    c.name = "crossings"
                    crossings[identity] = c

                    f = obj.create_value()
                    f.name = "path fragments"
                    fragments[identity] = f

                    m = obj.create_value()
                    m.name = "modifications"
                    modifications[identity] = m

                    i = obj.create_value()
                    i.name = "switch identities"
                    idswitchs[identity] = i

                    obj.idtrackerai_path = path
                    path.contours = cnt
                    path.crossings = c
                    path.fragments = f
                    path.modifications = m
                    path.switch_identity = i

                centroid = (
                    (
                        int(round(centroid[0] / resolution)),
                        int(round(centroid[1] / resolution)),
                    )
                    if centroid is not None
                    else None
                )

                paths[identity].contours.set_contour(
                    frame_index, np.int32(np.rint(contour / resolution))
                )
                paths[identity][frame_index] = centroid
                crossings[identity][frame_index] = 1 if crossing else 0
                fragments[identity][frame_index] = fragment

    # update the progress
    if progress_event is not None:
        progress_event(len(b.blobs_in_video), max_count=total_blobs)
