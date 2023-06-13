from idtrackerai import ListOfBlobs, Video
from idtrackerai.utils import create_dir

from .crossing_detector import detect_crossings
from .model_area import compute_body_length


def crossings_detection_API(video: Video, list_of_blobs: ListOfBlobs) -> None:
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
    video.crossing_detector_timer.start()

    median_body_length = compute_body_length(list_of_blobs, video.number_of_animals)
    video.set_id_image_size(median_body_length)

    create_dir(video.id_images_folder, remove_existing=True)

    list_of_blobs.set_images_for_identification(
        video.episodes,
        video.id_images_file_paths,
        video.id_image_size,
        video.segmentation_data_folder,
    )
    list_of_blobs.compute_overlapping_between_subsequent_frames()

    if video.single_animal:
        for blob in list_of_blobs.all_blobs:
            blob.is_an_individual = True
    else:
        detect_crossings(list_of_blobs, video)

    video.crossing_detector_timer.finish()


__all__ = ["crossings_detection_API"]
