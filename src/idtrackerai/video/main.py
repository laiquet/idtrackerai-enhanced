import argparse
import logging
from pathlib import Path

import numpy as np

from idtrackerai import Session
from idtrackerai.utils import IdtrackeraiError, wrap_entrypoint

from .general_video import generate_trajectories_video
from .individual_videos import generate_individual_video


@wrap_entrypoint
def main():
    # TODO clean up argparser, add subparsers

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
            "session_dir/trajectories/without_gaps.npy"
        ),
        metavar="",
    )
    parser.add_argument(
        "--tl",
        type=int,
        default=20,
        help=(
            "Trail length, number of points used to draw the individual trajectories"
            " traces in general videos. Default is 20"
        ),
        metavar="",
    )
    parser.add_argument(
        "--s", type=int, default=0, help="Frame where to start the video", metavar=""
    )
    parser.add_argument(
        "--e", type=int, help="Frame where to end the video", metavar=""
    )
    parser.add_argument(
        "--size",
        type=int,
        help=(
            "Size of the squared individual videos. Defaults to the median body length"
            " of the animals."
        ),
        metavar="",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Show centroids in general video without labeled ID's",
    )
    args = parser.parse_args()

    try:
        session = Session.load(args.session_path)
    except FileNotFoundError as exc:
        raise IdtrackeraiError() from exc

    if args.t is None:
        possible_files = (
            "validated.npy",
            "without_gaps.npy",
            "with_gaps.npy",
            "trajectories_validated.npy",
            "trajectories_wo_gaps.npy",
            "trajectories.npy",
            "trajectories_wo_identification.npy",
        )
        for file in possible_files:
            path = session.trajectories_folder / file
            if path.is_file():
                logging.info("Loading trajectories from %s", path)
                trajectories = np.load(path, allow_pickle=True).item()["trajectories"]
                break
        else:
            raise FileNotFoundError(
                f"Could not find the trajectory file in {session.trajectories_folder}"
            )
    else:
        logging.info("Loading trajectories from %s", args.t)
        trajectories = np.load(args.t, allow_pickle=True).item()["trajectories"]

    if args.individual:
        if args.no_labels:
            logging.info('Ignoring "--no-labels" flag')

        generate_individual_video(
            session,
            trajectories,
            draw_in_gray=args.gray,
            starting_frame=args.s,
            ending_frame=args.e,
            miniframe_size=args.size,
        )
    else:
        generate_trajectories_video(
            session,
            trajectories,
            draw_in_gray=args.gray,
            centroid_trace_length=args.tl,
            starting_frame=args.s,
            ending_frame=args.e,
            no_labels=args.no_labels,
        )


if __name__ == "__main__":
    main()
