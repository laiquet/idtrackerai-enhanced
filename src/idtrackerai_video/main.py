import logging
from pathlib import Path

import numpy as np

from idtrackerai import Video
from idtrackerai.utils import initLogger, wrap_exceptions

from .general_video import generate_trajectories_video
from .individual_videos import generate_individual_video


@wrap_exceptions
def main():
    initLogger(check_version=False)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "session_path",
        type=Path,
        help="Path to the video session created during the tracking session",
        metavar="",
    )

    parser.add_argument(
        "--individual",
        action="store_true",
        help="Generate individual video. Default is a general video",
    )
    parser.add_argument(
        "--gray", action="store_true", help="Draw the original video in grayscale"
    )

    parser.add_argument(
        "--t",
        type=Path,
        help=(
            "Path to the trajectory file, default is "
            "session_dir/trajectories/trajectories_wo_gaps.npy"
        ),
        metavar="",
    )
    parser.add_argument(
        "--tl",
        type=int,
        default=20,
        help=(
            "Trail length, number of points used to draw the individual trajectories"
            " traces"
        ),
        metavar="",
    )
    parser.add_argument(
        "--s", type=int, default=0, help="Frame where to start the video", metavar=""
    )
    parser.add_argument(
        "--e", type=int, help="Frame where to end the video", metavar=""
    )
    args = parser.parse_args()

    video = Video.load(args.session_path)

    if args.t is None:
        possible_files = (
            "trajectories_validated.npy",
            "trajectories_wo_gaps.npy",
            "trajectories.npy",
            "trajectories_wo_identification.npy",
        )
        for file in possible_files:
            path = video.trajectories_folder / file
            if path.is_file():
                logging.info("Loading trajectories from %s", path)
                trajectories = np.load(path, allow_pickle=True).item()["trajectories"]
                break
        else:
            raise FileNotFoundError(
                f"Could not find the trajectory file in {video.trajectories_folder}"
            )
    else:
        logging.info("Loading trajectories from %s", args.t)
        trajectories = np.load(args.t, allow_pickle=True).item()["trajectories"]

    if args.individual:
        generate_individual_video(
            video,
            trajectories,
            draw_in_gray=args.gray,
            starting_frame=args.s,
            ending_frame=args.e,
        )
    else:
        generate_trajectories_video(
            video,
            trajectories,
            draw_in_gray=args.gray,
            centroid_trace_length=args.tl,
            starting_frame=args.s,
            ending_frame=args.e,
        )


if __name__ == "__main__":
    main()
