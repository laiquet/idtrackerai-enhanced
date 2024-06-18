import logging

import numpy as np

from idtrackerai import ListOfFragments, ListOfGlobalFragments, Session
from idtrackerai.utils import IdtrackeraiError, conf, create_dir

from ..network import CNN, DEVICE, LearnerClassification, NetworkParams
from .accumulation_manager import AccumulationManager
from .accumulator import perform_one_accumulation_step
from .assigner import assign_remaining_fragments
from .identity_transfer import identify_first_global_fragment_for_accumulation
from .pre_trainer import pretrain_global_fragment


class TrackerAPI:
    "API for tracking with identities more than one animal with more than one Global Fragment"

    identification_model: CNN
    accumulation_network_params: NetworkParams

    def __init__(
        self,
        session: Session,
        list_of_fragments: ListOfFragments,
        list_of_global_fragments: ListOfGlobalFragments,
    ):
        self.session = session
        self.list_of_fragments = list_of_fragments
        self.list_of_global_fragments = list_of_global_fragments

    def track(self) -> ListOfFragments:
        """In protocol 3, list_of_fragments is loaded from accumulation
        folders so the reference from outside tracker_API is lost.
        That's why list_of_fragments has to be returned"""
        logging.info("Tracking with identities")
        self.session.accumulation_trial = 0
        create_dir(self.session.accumulation_folder, remove_existing=True)
        self.accumulation_network_params = NetworkParams(
            n_classes=self.session.n_animals,
            save_folder=self.session.accumulation_folder,
            knowledge_transfer_folder=self.session.knowledge_transfer_folder,
            model_name="identification_network",
            image_size=self.session.id_image_size,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION, "momentum": 0.9},
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
        )
        self.accumulation_network_params.save()
        with self.session.new_timer("Accumulation"):
            self.accumulation_protocol()
        with self.session.new_timer("Identification"):
            assign_remaining_fragments(
                self.list_of_fragments,
                self.identification_model,
                self.accumulation_network_params,
            )
        return self.list_of_fragments

    def accumulation_protocol(self) -> None:

        self.list_of_fragments.reset(roll_back_to="fragmentation")

        if self.session.knowledge_transfer_folder:
            try:
                self.identification_model = LearnerClassification.load_model(
                    self.accumulation_network_params, knowledge_transfer=True
                )
                logging.info("Tracking with knowledge transfer")
                if not self.session.identity_transfer:
                    self.identification_model.fully_connected_reinitialization()
                else:
                    logging.info(
                        "Identity transfer. Not reinitializing the fully connected"
                        " layers."
                    )
            except RuntimeError as exc:
                logging.error(
                    f"Could not load model {self.accumulation_network_params} to"
                    " transfer knowledge from, following without knowledge nor identity"
                    " transfer.\n"
                    f"Raised error: {exc}"
                )
                self.identification_model = CNN.from_network_params(
                    self.accumulation_network_params
                ).to(DEVICE)
        else:
            self.identification_model = CNN.from_network_params(
                self.accumulation_network_params
            ).to(DEVICE)

        first_global_fragment = max(
            self.list_of_global_fragments, key=lambda gf: gf.minimum_distance_travelled
        )

        self.session.first_frame_first_global_fragment.append(
            first_global_fragment.first_frame_of_the_core
        )

        identify_first_global_fragment_for_accumulation(
            first_global_fragment,
            self.session,
            identification_model=self.identification_model,
        )

        self.session.identities_groups = self.list_of_fragments.build_exclusive_rois()

        # Order global fragments by distance to the first global fragment for the accumulation
        self.list_of_global_fragments.sort_by_distance_to_the_frame(
            first_global_fragment.first_frame_of_the_core
        )

        # Instantiate accumulation manager
        self.accumulation_manager = AccumulationManager(
            self.session.n_animals,
            self.list_of_fragments,
            self.list_of_global_fragments,
        )

        # Selecting the first global fragment is considered as
        # the 0 accumulation step
        success = self.accumulate()

        self.save_after_first_accumulation()

        if success:
            return

        logging.warning(
            "[red]Protocol 2 failed, protocol 3 is going to start",
            extra={"markup": True},
        )
        ask_about_protocol3(
            self.session.protocol3_action, self.session.number_of_error_frames
        )

        with self.session.new_timer("Protocol 3 pre-training"):
            self.pretrain()

        with self.session.new_timer("Protocol 3 accumulation"):
            for self.session.accumulation_trial in range(
                1, conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS + 1
            ):
                try:
                    self.accumulation_parachute_init(self.session.accumulation_trial)
                except IndexError:
                    logging.warning(
                        "There are no more Global Fragments to start new accumulations"
                    )
                    break

                success = self.accumulate()
                self.save_and_update_accumulation_parameters_in_parachute()
                if success:
                    logging.info("Accumulation after protocol 3 has been successful")
                    break
                logging.warning("Accumulation after protocol 3 failed")
            else:
                logging.warning(
                    "All accumulation trials after after Protocol 3 pretrain failed"
                )

            self.load_best_accumulation()

    def accumulate(self) -> bool:
        while self.accumulation_manager.new_global_fragments_for_training:
            early_stopped = perform_one_accumulation_step(
                self.accumulation_manager,
                self.session,
                self.identification_model,
                self.accumulation_network_params,
            )
            if early_stopped:
                logging.info("We don't need to accumulate more images")
                break
        else:
            logging.info("No more new images to accumulate")

        if (
            self.accumulation_manager.ratio_accumulated_images
            > conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
        ):
            logging.info("We accumulated enough images")
            return True
        logging.info("[red]We did not accumulate enough images", extra={"markup": True})
        return False

    def save_after_first_accumulation(self):
        """Set flags and save data"""
        logging.info("Saving first accumulation parameters")

        # if not self.restoring_first_accumulation:
        self.session.ratio_accumulated_images = (
            self.accumulation_manager.ratio_accumulated_images
        )
        self.session.percentage_of_accumulated_images = [
            self.session.ratio_accumulated_images
        ]
        self.session.save()
        self.list_of_fragments.save(self.session.fragments_path)
        self.list_of_fragments.save(self.session.accumulation_folder)
        self.list_of_global_fragments.save(self.session.global_fragments_path)

    def pretrain(self):
        create_dir(self.session.pretraining_folder, remove_existing=True)

        pretrain_network_params = NetworkParams(
            n_classes=self.session.n_animals,
            save_folder=self.session.pretraining_folder,
            model_name="identification_network",
            image_size=self.session.id_image_size,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION, "momentum": 0.9},
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            knowledge_transfer_folder=self.session.knowledge_transfer_folder,
        )
        pretrain_network_params.save()

        # Initialize network
        if pretrain_network_params.knowledge_transfer_folder:
            self.identification_model = LearnerClassification.load_model(
                pretrain_network_params, knowledge_transfer=True
            )
            self.identification_model.fully_connected_reinitialization()
        else:
            self.identification_model = CNN.from_network_params(
                pretrain_network_params
            ).to(DEVICE)

        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.sort_by_distance_travelled()

        pretraining_counter = -1
        ratio_of_pretrained_images = 0.0
        while ratio_of_pretrained_images < conf.MAX_RATIO_OF_PRETRAINED_IMAGES:
            pretraining_counter += 1
            logging.info(
                "[bold]New pretraining iteration[/], using the #%s global fragment",
                pretraining_counter,
                extra={"markup": True},
            )
            pretrain_global_fragment(
                self.identification_model,
                pretrain_network_params,
                self.list_of_global_fragments.global_fragments[pretraining_counter],
                self.session.id_images_file_paths,
            )
            ratio_of_pretrained_images = (
                self.list_of_fragments.ratio_of_images_used_for_pretraining
            )

            logging.debug(
                f"{ratio_of_pretrained_images:.2%} of the images have been used during"
                " pretraining (if higher than"
                f" {conf.MAX_RATIO_OF_PRETRAINED_IMAGES:.2%} we stop pretraining)"
            )

    def accumulation_parachute_init(self, iteration_number: int) -> None:
        logging.info("Starting parachute accumulation %i", iteration_number)
        logging.info(
            "Setting #%d global fragment for accumulation", iteration_number - 1
        )

        self.list_of_global_fragments.sort_by_distance_travelled()
        first_global_fragment = self.list_of_global_fragments.global_fragments[
            iteration_number - 1
        ]

        create_dir(self.session.accumulation_folder, remove_existing=True)
        self.list_of_fragments.reset(roll_back_to="fragmentation")

        self.session.first_frame_first_global_fragment.append(
            first_global_fragment.first_frame_of_the_core
        )

        identify_first_global_fragment_for_accumulation(
            first_global_fragment,
            self.session,
            (
                LearnerClassification.load_model(self.accumulation_network_params)
                if self.session.identity_transfer
                else None
            ),
        )
        self.session.identities_groups = self.list_of_fragments.build_exclusive_rois()

        # Sort global fragments by distance
        self.list_of_global_fragments.sort_by_distance_to_the_frame(
            self.session.first_frame_first_global_fragment[iteration_number - 1]
        )
        logging.info(
            "First frame of first Global Fragment %s",
            self.session.first_frame_first_global_fragment,
        )
        logging.info(
            "We will restore the network from a previous pretraining: %s",
            self.session.pretraining_folder,
        )

        # Set saving folders
        self.accumulation_network_params.save_folder = self.session.accumulation_folder

        # Set restoring model_file
        self.accumulation_network_params.restore_folder = (
            self.session.pretraining_folder
        )

        self.identification_model = LearnerClassification.load_model(
            self.accumulation_network_params
        )

        self.identification_model.fully_connected_reinitialization()

        # Instantiate accumualtion manager
        self.accumulation_manager = AccumulationManager(
            self.session.n_animals,
            self.list_of_fragments,
            self.list_of_global_fragments,
        )

        logging.info("Start accumulation")

    def save_and_update_accumulation_parameters_in_parachute(self) -> None:
        logging.info(
            "Accumulated images"
            f" {self.accumulation_manager.ratio_accumulated_images:.2%}"
        )
        self.session.ratio_accumulated_images = (
            self.accumulation_manager.ratio_accumulated_images
        )
        self.session.percentage_of_accumulated_images.append(
            self.session.ratio_accumulated_images
        )
        self.list_of_fragments.save(
            self.session.accumulation_folder / "list_of_fragments.json"
        )

    def load_best_accumulation(self) -> None:
        logging.info("Saving second accumulation parameters")

        # Choose best accumulation
        self.session.accumulation_trial = int(
            np.argmax(self.session.percentage_of_accumulated_images)
        )
        logging.info(f"Best accumulation is #{self.session.accumulation_trial}")

        # Update ratio of accumulated images and  accumulation folder
        self.session.ratio_accumulated_images = (
            self.session.percentage_of_accumulated_images[
                self.session.accumulation_trial
            ]
        )

        # Load list of fragments of the best accumulation
        self.list_of_fragments = ListOfFragments.load(
            self.session.accumulation_folder / "list_of_fragments.json"
        )
        self.list_of_fragments.save(self.session.fragments_path)
        self.list_of_global_fragments.save(self.session.global_fragments_path)

        # set restoring folder
        logging.info("Restoring networks to best second accumulation")
        self.accumulation_network_params.restore_folder = (
            self.session.accumulation_folder
        )

        # Load pretrained network
        self.identification_model = LearnerClassification.load_model(
            self.accumulation_network_params
        )

        self.session.save()


def ask_about_protocol3(protocol3_action: str, n_error_frames: int) -> None:
    """Raises a IdtrackeraiError if protocol3_action is abort or aks and user answers abortion"""
    logging.info("Protocol 3 action: %s", protocol3_action)

    if protocol3_action == "abort":
        raise IdtrackeraiError(
            "Protocol 3 was going to start but PROTOCOL3_ACTION is set to 'abort'"
        )
    if protocol3_action == "continue":
        return

    if protocol3_action != "ask":
        raise ValueError(
            f'PROTOCOL3_ACTION "{protocol3_action}" not in ("ask", "abort", "continue")'
        )

    if n_error_frames > 0:
        logging.info(
            "Protocol 3 is a very time consuming algorithm and, in most cases, it"
            " can be avoided by redefining the segmentation parameters. As"
            " [red]there are %d frames with more blobs than animals[/red], we"
            " recommend you to abort the tracking session now and go back to the"
            " Segmentation app focusing on not having reflections, shades, etc."
            " detected as blobs. Check the following general recommendations:\n   "
            " - Define a region of interest to exclude undesired noise blobs\n    -"
            " Shrink the intensity (or background difference) thresholds\n    -"
            " Toggle the use of the background subtraction\n    - Shrink the blob's"
            " area thresholds",
            n_error_frames,
            extra={"markup": True},
        )
    else:
        logging.info(
            "Protocol 3 is a very time consuming algorithm and, in most cases, it"
            " can be avoided by redefining the segmentation parameters. As"
            " [bold]there are NOT frames with more blobs than animals[/bold], the"
            " video is unlikely to have non-animal blobs. Even so, you can choose"
            " to abort the tracking session and redefine the segmentation"
            " parameters (specially shrinking the intensity (or background"
            " difference) thresholds) or to continue with Protocol 3.",
            extra={"markup": True},
        )

    abort = None
    valid_answers = {"abort": True, "a": True, "continue": False, "c": False}
    while abort is None:
        answer_str = input(
            "What do you want to do now? Abort [A] or Continue [C]? "
        ).lower()
        if answer_str not in valid_answers:
            logging.warning("Invalid answer")
            continue
        abort = valid_answers[answer_str]
        logging.info("Answer --> Abort? %s", abort)
    if abort:
        raise IdtrackeraiError(
            "This is not an actual error: protocol 3 was going to start"
            " but PROTOCOL3_ACTION is set to 'ask' and used aborted."
        )
    return
