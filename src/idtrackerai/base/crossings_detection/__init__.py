from idtrackerai import ListOfBlobs, Session
from idtrackerai.utils import create_dir

from .crossing_detector import detect_crossings
from .model_area import compute_body_length


def crossings_detection_API(session: Session, list_of_blobs: ListOfBlobs) -> None:
    """
    This crossings detector works under the following assumptions
        1. The number of animals in the video is known (given by the user)
        2. There are frames in the video where all animals are separated from
        each other.
        3. All animals have a similar size
        4. The frame rate of the video is higher enough so that consecutive
        segmented blobs of pixels of the same animal overlap, i.e. some of the
        pixels representing the animal A in frame i are the same in the
        frame i+1.

    NOTE: This crossing detector sets the identification images that will be
    used to identify the animals
    """
    session.crossing_detector_timer.start()

    median_body_length = compute_body_length(list_of_blobs, session.n_animals)
    session.set_id_image_size(median_body_length)

    create_dir(session.id_images_folder, remove_existing=True)

    list_of_blobs.set_images_for_identification(
        session.episodes,
        session.id_images_file_paths,
        session.id_image_size,
        session.number_of_parallel_workers,
    )
    list_of_blobs.compute_overlapping_between_subsequent_frames()

    if session.single_animal:
        for blob in list_of_blobs.all_blobs:
            blob.is_an_individual = True
    else:
        detect_crossings(list_of_blobs, session)

    session.crossing_detector_timer.finish()


__all__ = ["crossings_detection_API"]
