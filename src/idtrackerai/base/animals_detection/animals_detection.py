import logging

import cv2

from idtrackerai import ListOfBlobs, Session
from idtrackerai.utils import IdtrackeraiError, create_dir, remove_dir

from .segmentation import compute_background, segment


def animals_detection_API(session: Session):
    """
    This class generates a ListOfBlobs object and updates the Session
    object with information about the process.
    """
    session.detect_animals_timer.start()
    if session.bounding_box_images_on_ram:
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

    detection_parameters = {
        "intensity_ths": session.intensity_ths,
        "area_ths": session.area_ths,
        "ROI_mask": session.ROI_mask,
        "bkg_model": bkg_model,
        "resolution_reduction": session.resolution_reduction,
    }

    if session.resolution_reduction != 1 and bkg_model is not None:
        detection_parameters["bkg_model"] = cv2.resize(
            bkg_model,
            None,  # type: ignore
            fx=session.resolution_reduction,
            fy=session.resolution_reduction,
            interpolation=cv2.INTER_AREA,
        )

    # Main call
    blobs_in_video = segment(
        detection_parameters,
        session.episodes,
        (None if session.bounding_box_images_on_ram else session.bbox_images_folder),
        session.number_of_frames,
        session.number_of_parallel_workers,
    )

    list_of_blobs = ListOfBlobs(blobs_in_video)
    assert len(list_of_blobs) == session.number_of_frames
    logging.info(f"{list_of_blobs.number_of_blobs} detected blobs in total")

    if session.n_animals > 0:
        check_segmentation(session, list_of_blobs)

    session.detect_animals_timer.finish()
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
    n_frames_with_all_visible = sum(
        n_blobs_in_frame == session.n_animals
        for n_blobs_in_frame in map(len, list_of_blobs.blobs_in_video)
    )

    if n_frames_with_all_visible == 0:
        raise IdtrackeraiError(
            "There is no frames where the number of blobs is equal "
            "to the number of animals stated by the user. Idtracker.ai "
            "needs those frame to work"
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
