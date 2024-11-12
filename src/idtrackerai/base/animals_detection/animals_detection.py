import logging

from idtrackerai import IdtrackeraiError, ListOfBlobs, Session
from idtrackerai.utils import create_dir, remove_dir

from .segmentation import compute_background, segment


def animals_detection_API(session: Session) -> ListOfBlobs:
    """
    This class generates a ListOfBlobs object and updates the Session
    object with information about the process.
    """
    if session.bounding_box_images_in_ram:
        remove_dir(session.bbox_images_folder)
    else:
        create_dir(session.bbox_images_folder, remove_existing=True)

    bkg_model = session.bkg_model
    if session.use_bkg:
        if bkg_model is None:
            bkg_model = compute_background(
                session.episodes,
                session.number_of_frames_for_background,
                session.background_subtraction_stat,
            )
            session.bkg_model = bkg_model
        else:
            logging.info("Using previously computed background model from GUI")
    else:
        bkg_model = None
        logging.info("No background model computed")

    # Main call
    blobs_in_video = segment(
        {
            "intensity_ths": session.intensity_ths,
            "area_ths": session.area_ths,
            "ROI_mask": session.ROI_mask,
            "bkg_model": bkg_model,
        },
        session.episodes,
        None if session.bounding_box_images_in_ram else session.bbox_images_folder,
        session.number_of_frames,
        session.number_of_parallel_workers,
    )

    list_of_blobs = ListOfBlobs(blobs_in_video)
    assert len(list_of_blobs) == session.number_of_frames

    trackable_frames = (
        sum(end - start for start, end in session.tracking_intervals)
        if session.tracking_intervals
        else session.number_of_frames
    )
    n_detected_blobs = list_of_blobs.number_of_blobs
    logging.info(
        f"{n_detected_blobs} detected blobs in total, an average of {n_detected_blobs / trackable_frames:.1f} blobs per frame"
    )

    if session.n_animals > 0:
        check_segmentation(session, list_of_blobs)

    return list_of_blobs


def check_segmentation(session: Session, list_of_blobs: ListOfBlobs):
    """
    idtracker.ai is designed to work under the assumption that all the
    detected blobs are animals. In the frames where the number of
    detected blobs is higher than the number of animals in the video, it is
    likely that some blobs do not represent animals. In this scenario
    idtracker.ai might misbehave. This method allows to check such
    condition.
    """

    if not list_of_blobs.number_of_blobs:
        raise IdtrackeraiError("No animals detected in the video")

    n_frames_with_all_visible = sum(
        n_blobs_in_frame == session.n_animals
        for n_blobs_in_frame in map(len, list_of_blobs.blobs_in_video)
    )

    if n_frames_with_all_visible == 0:
        logging.warning(
            "There are no frames where the number of blobs is equal "
            f"to the number of animals stated by the user {session.n_animals}. "
            "Depending on the video this is inevitable and idtracker.ai can handle it. "
            "But, if possible, check the segmentation parameters to make sure "
            "your animals are visible and properly segmented."
        )

    error_frames = [
        frame
        for frame, blobs in enumerate(list_of_blobs.blobs_in_video)
        if len(blobs) > session.n_animals
    ]

    n_error_frames = len(error_frames)
    logging.log(
        logging.WARNING if n_error_frames else logging.INFO,
        f"There are {n_error_frames} frames with more blobs than animals",
    )
    session.number_of_error_frames = n_error_frames

    output_path = session.session_folder / "inconsistent_frames.csv"
    output_path.unlink(missing_ok=True)

    if not n_error_frames:
        return

    logging.warning("This can be detrimental for the proper functioning of the system")
    if n_error_frames < 25:
        logging.warning(f"Frames with more blobs than animals: {error_frames}")
    else:
        logging.warning(
            "Too many frames with more blobs than animals "
            "for printing their indices in log"
        )

    logging.info(
        f"Saving indices of frames with more blobs than animals in {output_path}"
    )
    output_path.write_text("\n".join(map(str, error_frames)))

    if session.check_segmentation:
        list_of_blobs.save(session.blobs_path)
        raise IdtrackeraiError(
            f"Check_segmentation is {True}, exiting...\n"
            "Please readjust the segmentation parameters and track again"
        )
    logging.info(f"Check_segmentation is {False}, ignoring the above errors")
