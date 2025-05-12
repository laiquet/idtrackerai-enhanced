import json
import logging
from pathlib import Path

from idtrackerai import (
    GlobalFragment,
    ListOfFragments,
    ListOfGlobalFragments,
    Session,
    __version__,
    conf,
)
from idtrackerai.utils import create_dir

from ..network import DEVICE, IdCNN, IdentifierBase, IdentifierIdCNN
from .accumulation_manager import AccumulationManager
from .accumulator import accumulation_step
from .assigner import assign_remaining_fragments, check_penultimate_model
from .contrastive import ContrastiveLearning, IdentifierContrastive
from .identity_transfer import identify_first_global_fragment_for_accumulation


def run_tracker(
    session: Session,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
) -> IdentifierBase:
    logging.info("Tracking with identities")
    create_dir(session.accumulation_folder, remove_existing=True)

    with session.new_timer("Fragment identification"):
        identifier_model = fragment_identification(
            session, list_of_fragments, list_of_global_fragments
        )
        identifier_model.save(session.accumulation_folder)

    with (session.accumulation_folder / "model_params.json").open("w") as file:
        json.dump(
            {
                "n_classes": session.n_animals,
                "image_size": session.id_image_size,
                "resolution_reduction": session.resolution_reduction,
                "model": identifier_model.model.__class__.__name__,
                "version": __version__,
            },
            file,
            indent=4,
        )

    with session.new_timer("Identification"):
        assign_remaining_fragments(list_of_fragments, identifier_model)

    return identifier_model


def fragment_identification(
    session: Session,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
) -> IdentifierBase:

    list_of_fragments.reset(roll_back_to="fragmentation")

    # Instantiate accumulation manager
    accumulation_manager = AccumulationManager(
        session.n_animals, list_of_fragments, list_of_global_fragments
    )

    first_global_fragment = (
        max(list_of_global_fragments, key=lambda gf: gf.minimum_distance_travelled)
        if list_of_global_fragments.global_fragments
        else None
    )

    if first_global_fragment is None:
        logging.info("The video does not contain any long enough Global Fragment")
        if session.exclusive_rois:
            logging.warning(  # TODO
                "Right now it is not possible to have exclusive ROIs without Global Fragments. We are working on it"
            )
    else:
        session.first_frame_first_global_fragment = (
            first_global_fragment.first_frame_of_the_core
        )
        identify_first_global_fragment_for_accumulation(
            first_global_fragment,
            session,
            session.knowledge_transfer_folder,
            session.id_image_size,
        )
        session.identities_groups = list_of_fragments.build_exclusive_rois()

    list_of_global_fragments.sort_by_distance_to_the_frame(
        session.first_frame_first_global_fragment
    )

    if conf.DISABLE_CONTRASTIVE:
        logging.warning("Contrastive step is disabled")
    else:
        with session.new_timer("Contrastive step"):
            identifier_contrastive, ratio_accumulated = contrastive_step(
                first_global_fragment,
                session.knowledge_transfer_folder,
                list_of_fragments,
                session,
                accumulation_manager,
            )
            session.ratio_accumulated_images = ratio_accumulated

            if ratio_accumulated == float("inf"):
                logging.info(
                    "There are no Global Fragments for an accumulation protocol\n"
                    "[bold]We will not run the accumulation protocol[/].",
                    extra={"markup": True},
                )
                return identifier_contrastive

            logging.info(
                f"Contrastive step identified {ratio_accumulated:.2%} of the accumulable images"
            )
            THRESHOLD_EARLY_STOP_ACCUMULATION = conf.THRESHOLD_EARLY_STOP_ACCUMULATION
            if ratio_accumulated >= THRESHOLD_EARLY_STOP_ACCUMULATION:
                logging.info(
                    f"This is higher than {THRESHOLD_EARLY_STOP_ACCUMULATION=:.1%}, "
                    "enough to finish accumulation right here.\n"
                    "[bold]We will not run the accumulation protocol[/].",
                    extra={"markup": True},
                )

                # if we do not run the accumulation protocol, we need to set the final
                # identities to the fragments accumulated by contrastive
                for frag in list_of_fragments:
                    if frag.acceptable_for_training and not frag.used_for_training:
                        assert frag.temporary_id is not None
                        frag.used_for_training = True
                        frag.acceptable_for_training = False
                        frag.accumulated_globally = True
                        frag.accumulation_step = 0
                        frag.identity = frag.temporary_id + 1
                        frag.P1_vector[:] = 0.0
                        frag.P1_vector[frag.temporary_id] = 1.0

                return identifier_contrastive

            logging.info(
                f"This is lower than {THRESHOLD_EARLY_STOP_ACCUMULATION=:.1%}, "
                "[bold]not[/] enough to finish accumulation right here.\n"
                "[bold]We will run the accumulation protocol[/].",
                extra={"markup": True},
            )

            if ratio_accumulated < 0.8 and session.silhouette_score is not None:
                if session.silhouette_score > conf.CONTRASTIVE_SILHOUETTE_TARGET:
                    logging.warning(
                        "Such a low ratio of accumulated images with a Silhouette score "
                        "above the target may indicate the need to increase such target "
                        "or to check again the segmentation parameters"
                    )
                else:
                    logging.warning(
                        "Such a low ratio of accumulated images with a Silhouette score "
                        "below the target may indicate the need to increase the training "
                        "patience or to check again the segmentation parameters"
                    )

    if session.knowledge_transfer_folder:
        try:
            identification_cnn = IdCNN.load(
                session.id_image_size, session.knowledge_transfer_folder
            ).to(DEVICE)
        except FileNotFoundError:
            logging.warning(
                "IdCNN model not found in the knowledge transfer folder "
                f'"{session.knowledge_transfer_folder}", proceeding with a randomly initialized model'
            )
            identification_cnn = IdCNN(session.id_image_size, session.n_animals).to(
                DEVICE
            )
    else:
        identification_cnn = IdCNN(session.id_image_size, session.n_animals).to(DEVICE)

    if conf.TORCH_COMPILE:
        identification_cnn.compile()

    model_path = session.accumulation_folder / "tmp_identification_network.pt"
    penultimate_model_path = session.accumulation_folder / (
        "tmp_identification_network_penultimate.pt"
    )

    with session.new_timer("Accumulation protocol"):
        while accumulation_manager.new_global_fragments_for_training:
            early_stopped = accumulation_step(
                accumulation_manager,
                session,
                identification_cnn,
                model_path,
                penultimate_model_path,
            )
            if early_stopped:
                logging.info("We don't need to accumulate more images")
                break
        else:
            logging.info("No more new images to accumulate")

    ratio_accumulated = accumulation_manager.ratio_accumulated_images
    session.ratio_accumulated_images = ratio_accumulated

    if ratio_accumulated > 0.9:
        logging.info(
            f"[green]We accumulated enough images ({ratio_accumulated:.2%}). Accumulation protocol succeeded!",
            extra={"markup": True},
        )
    else:
        logging.warning(
            f"[red]We did not accumulate enough images ({ratio_accumulated:.2%}). Accumulation protocol failed!",
            extra={"markup": True},
        )

    check_penultimate_model(identification_cnn, model_path, penultimate_model_path)
    model_path.unlink(missing_ok=True)
    model_path.with_suffix(".metadata.json").unlink(missing_ok=True)
    penultimate_model_path.unlink(missing_ok=True)
    penultimate_model_path.with_suffix(".metadata.json").unlink(missing_ok=True)

    return IdentifierIdCNN(identification_cnn)


def contrastive_step(
    first_global_fragment: GlobalFragment | None,
    knowledge_transfer_folder: Path | None,
    list_of_fragments: ListOfFragments,
    session: Session,
    accumulation_manager: AccumulationManager,
) -> tuple[IdentifierContrastive, float]:
    contrastive = ContrastiveLearning(
        list_of_fragments,
        saving_folder=session.accumulation_folder,
        first_gfrag=first_global_fragment,
    )
    try:
        contrastive.set_model(knowledge_transfer_folder)
    except FileNotFoundError as exc:
        logging.error(exc)
        contrastive.set_model()

    session.silhouette_score = contrastive.train()
    contrastive.predict(list_of_fragments, first_global_fragment)

    if not list_of_fragments.n_images_in_global_fragments:
        # there are no global fragments
        contrastive.model_checkpoint_path.unlink()
        return contrastive.get_identification_model(), float("inf")

    accumulation_manager.assign_identities()
    accumulation_manager.update_accumulation_statistics()
    session.accumulation_statistics_data = accumulation_manager.accumulation_statistics

    n_accumulated_images = sum(
        fragment.n_images
        for fragment in list_of_fragments.individual_fragments
        if fragment.acceptable_for_training and not fragment.used_for_training
    )

    ratio = n_accumulated_images / list_of_fragments.n_images_in_global_fragments

    if ratio > conf.THRESHOLD_EARLY_STOP_ACCUMULATION:
        # remove contrastive checkpoint because the whole IdentifierContrastive will be saved instead
        contrastive.model_checkpoint_path.unlink(missing_ok=True)

    return contrastive.get_identification_model(), ratio
