from pathlib import Path

import numpy as np

from idtrackerai import Video
from idtrackerai_GUI_tools import initLogger

from . import generate_individual_video, generate_trajectories_video


def main():
    initLogger(check_version=False)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "session_path",
        type=Path,
        help="Path to the video session created during the tracking session",
    )

    parser.add_argument(
        "-gray", action="store_true", help="Draw the original video in grayscale"
    )

    parser.add_argument(
        "-t",
        "--trajectories_path",
        type=Path,
        help="Path to the trajectory file, default is session_dir/trajectories/trajectories*",
        default=None,
    )
    parser.add_argument(
        "-gp",
        "--number_of_ghost_points",
        type=int,
        default=20,
        help="Number of points used to draw the individual trajectories' traces",
    )
    parser.add_argument(
        "-sf",
        "--starting_frame",
        type=int,
        default=0,
        help="Frame where to start the video",
    )
    parser.add_argument(
        "-ef",
        "--ending_frame",
        type=int,
        default=None,
        help="Frame where to end the video",
    )
    parser.add_argument("-individual", action="store_true")
    args = parser.parse_args()

    video = Video.load(args.session_path)
    if args.trajectories_path is None:
        if (video.trajectories_folder / "trajectories_wo_gaps.npy").is_file():
            trajectories = np.load(
                video.trajectories_folder / "trajectories_wo_gaps.npy",
                allow_pickle=True,
            ).item()["trajectories"]
        elif (video.trajectories_folder / "trajectories.npy").is_file():
            trajectories = np.load(
                video.trajectories_folder / "trajectories.npy", allow_pickle=True
            ).item()["trajectories"]
        else:
            raise FileNotFoundError(
                f"Could not find the trajectory file in {video.trajectories_folder}"
            )
    else:
        trajectories = np.load(args.trajectories_path, allow_pickle=True).item()[
            "trajectories"
        ]
    if args.individual:
        generate_individual_video(
            video,
            trajectories,
            draw_in_gray=args.gray,
            centroid_trace_length=args.number_of_ghost_points,
            starting_frame=args.starting_frame,
            ending_frame=args.ending_frame,
        )
    else:
        generate_trajectories_video(
            video,
            trajectories,
            draw_in_gray=args.gray,
            centroid_trace_length=args.number_of_ghost_points,
            starting_frame=args.starting_frame,
            ending_frame=args.ending_frame,
        )
