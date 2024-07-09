import logging
from pathlib import Path

from idtrackerai import GlobalFragment, ListOfFragments, ListOfGlobalFragments, Session
from idtrackerai.utils import conf, create_dir

from ..network import CNN, DEVICE, IdentifierBase, IdentifierCNN, NetworkParams
from .accumulation_manager import AccumulationManager
from .accumulator import perform_one_accumulation_step
from .assigner import assign_remaining_fragments, check_penultimate_model
from .contrastive import ContrastiveLearning, IdentifierContrastive
from .identity_transfer import identify_first_global_fragment_for_accumulation


def run_tracker(
    session: Session,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
) -> None:
    "API for tracking with identities more than one animal with more than one Global Fragment"
    """In protocol 3, list_of_fragments is loaded from accumulation
        folders so the reference from outside tracker_API is lost.
        That's why list_of_fragments has to be returned"""
    logging.info("Tracking with identities")
    create_dir(session.accumulation_folder, remove_existing=True)
    accumulation_network_params = NetworkParams(
        n_classes=session.n_animals,
        save_folder=session.accumulation_folder,
        knowledge_transfer_folder=session.knowledge_transfer_folder,
        model_name="identification_network",
        image_size=session.id_image_size,
    )
    accumulation_network_params.save()
    with session.new_timer("Accumulation"):
        identifier_model, ratio_accumulated_images = accumulation_protocol(
            session,
            list_of_fragments,
            list_of_global_fragments,
            accumulation_network_params,
        )

    if isinstance(identifier_model, IdentifierCNN):
        check_penultimate_model(identifier_model.model, accumulation_network_params)
    identifier_model.save(session.accumulation_folder)
    session.ratio_accumulated_images = ratio_accumulated_images

    with session.new_timer("Identification"):
        assign_remaining_fragments(list_of_fragments, identifier_model)


def accumulation_protocol(
    session: Session,
    list_of_fragments: ListOfFragments,
    list_of_global_fragments: ListOfGlobalFragments,
    accumulation_network_params: NetworkParams,
) -> tuple[IdentifierBase, float]:

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
            identifier_contrastive, ratio = contrastive_step(
                first_global_fragment,
                session.knowledge_transfer_folder,
                list_of_fragments,
                session,
                accumulation_manager,
            )

            logging.info(
                f"Contrastive step identified {ratio:.2%} of the accumulable images"
            )

            if ratio >= conf.THRESHOLD_EARLY_STOP_ACCUMULATION:
                logging.info(
                    f"This is higher than {conf.THRESHOLD_EARLY_STOP_ACCUMULATION:.1%}, enough to finish accumulation right here.\n"
                    "[bold]We will not train the identifier CNN[/] and will use the contrastive clusters for the residual identification",
                    extra={"markup": True},
                )
                return identifier_contrastive, ratio
            else:
                logging.info(
                    f"This is lower than {conf.THRESHOLD_EARLY_STOP_ACCUMULATION:.1%}, [bold]not[/] enough to finish accumulation right here.\n"
                    "[bold]We will train the identifier CNN[/] in the following accumulations steps",
                    extra={"markup": True},
                )

    if session.knowledge_transfer_folder:
        identification_cnn = CNN.load(
            accumulation_network_params.image_size, session.knowledge_transfer_folder
        ).to(DEVICE)
    else:
        identification_cnn = CNN(
            accumulation_network_params.image_size,
            accumulation_network_params.n_classes,
        ).to(DEVICE)

    success, identifier_cnn = accumulate(
        accumulation_manager, session, accumulation_network_params, identification_cnn
    )

    if not success:
        logging.warning("[red]Protocol 2 failed", extra={"markup": True})
    return identifier_cnn, accumulation_manager.ratio_accumulated_images


def contrastive_step(
    first_global_fragment: GlobalFragment | None,
    knowledge_transfer_folder: Path | None,
    list_of_fragments: ListOfFragments,
    session: Session,
    accumulation_manager: AccumulationManager,
) -> tuple[IdentifierContrastive, float]:
    contrastive = ContrastiveLearning(
        list_of_fragments,
        session.accumulation_folder,
        check_every=max(5 * list_of_fragments.n_animals, 50),
        first_gfrag=first_global_fragment,
    )

    contrastive.set_model(knowledge_transfer_folder)
    contrastive.train()
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
        contrastive.model_checkpoint_path.unlink()

    return contrastive.get_identification_model(), ratio


def accumulate(
    accumulation_manager: AccumulationManager,
    session: Session,
    accumulation_network_params: NetworkParams,
    cnn: CNN,
) -> tuple[bool, IdentifierCNN]:
    while accumulation_manager.new_global_fragments_for_training:
        early_stopped = perform_one_accumulation_step(
            accumulation_manager, session, cnn, accumulation_network_params
        )
        if early_stopped:
            logging.info("We don't need to accumulate more images")
            break
    else:
        logging.info("No more new images to accumulate")

    if accumulation_manager.ratio_accumulated_images > 0.9:
        logging.info("We accumulated enough images")
        return True, IdentifierCNN(cnn)
    logging.info("[red]We did not accumulate enough images", extra={"markup": True})
    return False, IdentifierCNN(cnn)
