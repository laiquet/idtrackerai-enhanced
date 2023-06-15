# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)
import json
import logging

import numpy as np
import torch
from torch.backends import cudnn

from idtrackerai import ListOfBlobs, ListOfFragments, ListOfGlobalFragments, Video
from idtrackerai.network import (
    LearnerClassification,
    NetworkParams,
    fc_weights_reinit,
    get_device,
    weights_xavier_init,
)
from idtrackerai.utils import CustomError, conf, create_dir, json_object_hook

from .accumulation_manager import AccumulationManager
from .accumulator import perform_one_accumulation_step
from .assigner import assign_remaining_fragments
from .identity_transfer import identify_first_global_fragment_for_accumulation
from .pre_trainer import pre_train_global_fragment


class TrackerAPI:
    identification_model: torch.nn.Module

    def __init__(
        self,
        video: Video,
        list_of_blobs: ListOfBlobs,
        list_of_fragments: ListOfFragments,
        list_of_global_fragments: ListOfGlobalFragments,
    ):
        self.video = video
        self.list_of_blobs = list_of_blobs
        self.list_of_fragments = list_of_fragments
        self.list_of_global_fragments = list_of_global_fragments

        if self.video.knowledge_transfer_folder is not None:
            kt_info_dict_path = (
                self.video.knowledge_transfer_folder / "model_params.json"
            )
            try:
                self.knowledge_transfer_info_dict: dict = json.load(
                    kt_info_dict_path.open(), object_hook=json_object_hook
                )
            except FileNotFoundError:
                # Transferring from v4
                self.knowledge_transfer_info_dict: dict = np.load(
                    kt_info_dict_path.with_suffix(".npy"), allow_pickle=True
                ).item()
        else:
            self.knowledge_transfer_info_dict = {}

        # Old requirements for restoring
        # self.processes_to_restore = {}

        self.accumulation_network_params: NetworkParams
        # self.restoring_first_accumulation = False  # Flag restores first accumulation

    def track_single_animal(self):
        logging.debug("Assigning identity 1 to all blobs")
        for blob in self.list_of_blobs.all_blobs:
            blob.identity = 1

    def track_single_global_fragment_video(self):
        logging.info("TRACKING SINGLE GLOBAL FRAGMENT")

        fragment_identifier_to_id = {}
        identity = 1
        for fragment in self.list_of_fragments.fragments:
            if fragment.is_an_individual:
                fragment_identifier_to_id[fragment.identifier] = identity
                identity += 1
            else:
                fragment_identifier_to_id[fragment.identifier] = None

        for blob in (b for b in self.list_of_blobs.all_blobs if b.is_an_individual):
            blob.identity = fragment_identifier_to_id[blob.fragment_identifier]
        self.video.first_frame_first_global_fragment = [0]  # in case

    def track_with_identities(self) -> ListOfFragments:
        """In protocol 3, list_of_fragments is loaded from accumulation
        folders so the reference from outside tracker_API is lost.
        That's why list_of_fragments has to be returned"""
        self.video.tracking_timer.start()
        self.track_with_protocols_cascade()
        self.video.tracking_timer.finish()
        return self.list_of_fragments

    def track_with_protocols_cascade(self):
        logging.info("Starting protocol cascade")
        self.video.create_accumulation_folder(iteration_number=0, delete=True)
        self.accumulation_network_params = NetworkParams(
            number_of_classes=self.video.number_of_animals,
            architecture=conf.IDCNN_NETWORK_NAME,
            save_folder=self.video.accumulation_folder,
            knowledge_transfer_folder=self.video.knowledge_transfer_folder,
            model_name="identification_network",
            image_size=self.video.id_image_size,
            scopes_layers_to_optimize=conf.LAYERS_TO_OPTIMISE_PRETRAINING,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION, "momentum": 0.9},
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            return_store_objects=False,
        )
        self.accumulation_network_params.save()
        self.protocol1()

    def protocol1(self):
        self.video.protocol1_timer.start()

        # reset list of fragments and global fragments to fragmentation
        self.list_of_fragments.reset(roll_back_to="fragmentation")

        logging.info("Creating idCNN")
        if self.video.knowledge_transfer_folder:
            try:
                self.identification_model = LearnerClassification.load_model(
                    self.accumulation_network_params, knowledge_transfer=True
                )
                logging.info("Tracking with knowledge transfer")
                if not self.video.identity_transfer:
                    logging.info("Reinitializing fully connected layers")
                    self.identification_model.apply(fc_weights_reinit)
                else:
                    logging.info(
                        "Identity transfer. Not reinitializing the fully connected"
                        " layers."
                    )
            except RuntimeError:
                logging.error(
                    f"Could not load model {self.accumulation_network_params} to"
                    " transfer knowledge, following without knowledge nor identity"
                    " transfer"
                )
                self.identification_model = LearnerClassification.create_model(
                    self.accumulation_network_params
                )
                self.identification_model.apply(weights_xavier_init)
        else:
            self.identification_model = LearnerClassification.create_model(
                self.accumulation_network_params
            )
            self.identification_model.apply(weights_xavier_init)

        logging.info("Setting the first global fragment for accumulation")
        first_global_fragment = max(
            self.list_of_global_fragments.global_fragments,
            key=lambda gf: gf.minimum_distance_travelled,
        )

        self.video.first_frame_first_global_fragment.append(
            first_global_fragment.first_frame_of_the_core
        )

        identify_first_global_fragment_for_accumulation(
            first_global_fragment,
            self.video,
            network_params=self.accumulation_network_params,
            identification_model=self.identification_model,
            knowledge_transfer_info_dict=self.knowledge_transfer_info_dict,
        )

        # Order global fragments by distance to the first global fragment for the accumulation
        self.list_of_global_fragments.order_by_distance_to_the_frame(
            first_global_fragment.first_frame_of_the_core
        )

        # Instantiate accumulation manager
        self.accumulation_manager = AccumulationManager(
            self.video.id_images_file_paths,
            self.video.number_of_animals,
            self.list_of_fragments,
            self.list_of_global_fragments,
        )

        # Selecting the first global fragment is considered as
        # the 0 accumulation step
        self.video.init_accumulation_statistics_attributes()
        self.accumulate()

    def accumulate(self):
        logging.info("Entering accumulation loop")
        if self.accumulation_manager.new_global_fragments_for_training:
            # Training and identification continues
            if (
                self.accumulation_manager.current_step == 1
                and self.video.accumulation_trial == 0
            ):
                # first training finished
                self.video.protocol1_timer.finish()
                self.video.protocol2_timer.start()

            # Training and identification step
            perform_one_accumulation_step(
                self.accumulation_manager,
                self.video,
                self.identification_model,
                self.accumulation_network_params,
            )
            # Re-enter the function for the next step of the accumulation
            self.accumulate()

        elif (
            not self.video.protocol2_timer.finished
            and self.accumulation_manager.ratio_accumulated_images
            > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
        ):
            # Accumulation stop because protocol 1 is successful
            self.save_after_first_accumulation()
            self.video.protocol1_timer.finish()
            logging.info("Protocol 1 successful")
            assign_remaining_fragments(
                self.list_of_fragments,
                self.identification_model,
                self.accumulation_network_params,
                self.video.identify_timer,
            )

        elif not self.video.protocol3_pretraining_timer.finished:
            logging.info("No more new global fragments")
            self.save_after_first_accumulation()

            if (
                self.accumulation_manager.ratio_accumulated_images
                >= conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            ):
                self.video.protocol2_timer.finish()
                logging.info("Protocol 2 successful")
                assign_remaining_fragments(
                    self.list_of_fragments,
                    self.identification_model,
                    self.accumulation_network_params,
                    self.video.identify_timer,
                )

            else:
                self.video.protocol1_timer.finish()
                self.video.protocol2_timer.finish(raise_if_not_started=False)
                logging.warning(
                    "[red]Protocol 2 failed, protocol 3 is going to start",
                    extra={"markup": True},
                )
                ask_about_protocol3(
                    self.video.protocol3_action, self.video.number_of_error_frames
                )

                self.video.protocol3_pretraining_timer.start()

                self.pretraining_counter = 0
                self.protocol3()

        elif (
            self.video.protocol3_pretraining_timer.finished
            and self.video.accumulation_trial
            < conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
            and self.accumulation_manager.ratio_accumulated_images
            < conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
        ):
            logging.warning("Accumulation Protocol 3 failed. Opening parachute ...")
            if self.video.accumulation_trial == 0:
                self.video.protocol3_accumulation_timer.start()
            self.video.accumulation_trial += 1
            if self.video.accumulation_trial > 1:
                self.save_and_update_accumulation_parameters_in_parachute()
            self.accumulation_parachute_init(self.video.accumulation_trial)

            self.video.init_accumulation_statistics_attributes()
            self.accumulate()

        elif self.video.protocol3_pretraining_timer.finished and (
            self.accumulation_manager.ratio_accumulated_images
            >= conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            or self.video.accumulation_trial
            >= conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
        ):
            logging.info("Accumulation after protocol 3 has been successful")
            self.video.protocol3_accumulation_timer.finish()

            self.save_after_second_accumulation()
            assign_remaining_fragments(
                self.list_of_fragments,
                self.identification_model,
                self.accumulation_network_params,
                self.video.identify_timer,
            )

        # Whether to re-enter the function for the next accumulation step
        if self.accumulation_manager.new_global_fragments_for_training:
            self.accumulate()

    def save_after_first_accumulation(self):
        """Set flags and save data"""
        logging.info("Saving first accumulation parameters")

        # if not self.restoring_first_accumulation:
        self.video.ratio_accumulated_images = (
            self.accumulation_manager.ratio_accumulated_images
        )
        self.video.percentage_of_accumulated_images = [
            self.video.ratio_accumulated_images
        ]
        self.video.save()
        self.list_of_fragments.save(self.video.fragments_path)
        self.list_of_global_fragments.save(self.video.global_fragments_path)

    """ pretraining """

    def protocol3(self):
        self.init_pretraining_variables()

        logging.info(
            "Starting pretraining. Checkpoints will be stored in %s"
            % self.video.pretraining_folder
        )

        if self.video.knowledge_transfer_folder:
            logging.info(
                "Performing knowledge transfer from %s"
                % self.video.knowledge_transfer_folder
            )
            self.pretrain_network_params.knowledge_transfer_folder = (
                self.video.knowledge_transfer_folder
            )

        logging.info("Start pretraining")
        self.pretraining_step_finished = True
        self.pretraining_loop()

    def init_pretraining_variables(self):
        self.init_pretraining_net()
        self.ratio_of_pretrained_images = 0

        # Initialize network
        if self.video.knowledge_transfer_folder:
            self.identification_model = LearnerClassification.load_model(
                self.pretrain_network_params, knowledge_transfer=True
            )
            self.identification_model.apply(fc_weights_reinit)
        else:
            self.identification_model = LearnerClassification.create_model(
                self.pretrain_network_params
            )
            self.identification_model.apply(weights_xavier_init)

    def init_pretraining_net(self):
        create_dir(self.video.pretraining_folder, remove_existing=True)

        self.pretrain_network_params = NetworkParams(
            number_of_classes=self.video.number_of_animals,
            architecture=conf.IDCNN_NETWORK_NAME,
            save_folder=self.video.pretraining_folder,
            model_name="identification_network",
            image_size=self.video.id_image_size,
            scopes_layers_to_optimize=conf.LAYERS_TO_OPTIMISE_PRETRAINING,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION, "momentum": 0.9},
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            return_store_objects=False,
        )

    def pretraining_loop(self):
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.sort_by_distance_travelled()
        self.one_shot_pretraining()
        self.continue_pretraining()

    def one_shot_pretraining(self):
        self.pretraining_step_finished = False
        self.pretraining_global_fragment = (
            self.list_of_global_fragments.global_fragments[self.pretraining_counter]
        )
        (
            self.identification_model,
            self.ratio_of_pretrained_images,
            self.list_of_fragments,
            self.pretrained_model_path,
        ) = pre_train_global_fragment(
            self.video.number_of_animals,
            self.video.accumulation_step,
            self.identification_model,
            self.pretrain_network_params,
            self.pretraining_global_fragment,
            self.list_of_fragments,
        )
        self.pretraining_counter += 1
        self.pretraining_step_finished = True

    def continue_pretraining(self, clock_unschedule=None):
        if (
            self.pretraining_step_finished
            and self.ratio_of_pretrained_images < conf.MAX_RATIO_OF_PRETRAINED_IMAGES
        ):
            self.one_shot_pretraining()

            if clock_unschedule is None:
                self.continue_pretraining()

        elif self.ratio_of_pretrained_images > conf.MAX_RATIO_OF_PRETRAINED_IMAGES:
            logging.warning("Calling accumulate from continue_pretraining")
            self.video.protocol3_pretraining_timer.finish()
            self.accumulate()

    """ parachute """

    def accumulation_parachute_init(self, iteration_number: int):
        logging.debug("Accumulation_parachute_init")
        logging.info("Starting accumulation %i" % iteration_number)

        # delete = not self.processes_to_restore.get("protocol3_accumulation")

        self.video.create_accumulation_folder(
            iteration_number=iteration_number, delete=True
        )
        self.video.accumulation_trial = iteration_number
        self.list_of_fragments.reset(roll_back_to="fragmentation")

        logging.info(
            "Setting #%d global fragment for accumulation", iteration_number - 1
        )

        self.list_of_global_fragments.sort_by_distance_travelled()
        try:
            first_global_fragment = self.list_of_global_fragments.global_fragments[
                iteration_number - 1
            ]
        except IndexError:
            first_global_fragment = None  # TODO what if this happens

        self.video.first_frame_first_global_fragment.append(
            first_global_fragment.first_frame_of_the_core
            if first_global_fragment is not None
            else None
        )

        if first_global_fragment is not None:
            identify_first_global_fragment_for_accumulation(
                first_global_fragment,
                self.video,
                (
                    LearnerClassification.load_model(self.accumulation_network_params)
                    if self.video.identity_transfer
                    else None
                ),
                self.accumulation_network_params,
                self.knowledge_transfer_info_dict,
            )

        # Sort global fragments by distance
        self.list_of_global_fragments.order_by_distance_to_the_frame(
            self.video.first_frame_first_global_fragment[iteration_number - 1]
        )
        logging.warning(
            "first_frame_first_global_fragment "
            + str(self.video.first_frame_first_global_fragment)
        )
        logging.info(
            "We will restore the network from a previous pretraining: %s"
            % self.video.pretraining_folder
        )

        # Set saving folders
        self.accumulation_network_params.save_folder = self.video.accumulation_folder

        # Set restoring model_file
        self.accumulation_network_params.restore_folder = self.video.pretraining_folder

        # TODO: allow to train only the fully connected layers
        self.accumulation_network_params.scopes_layers_to_optimize = [
            "fully-connected1",
            "fully_connected_pre_softmax",
        ]
        logging.info("Initializing accumulation network")

        # Load pretrained network
        self.identification_model = LearnerClassification.load_model(
            self.accumulation_network_params
        )

        # Re-initialize fully-connected layers
        self.identification_model.apply(fc_weights_reinit)

        # Instantiate accumualtion manager
        self.accumulation_manager = AccumulationManager(
            self.video.id_images_file_paths,
            self.video.number_of_animals,
            self.list_of_fragments,
            self.list_of_global_fragments,
        )

        logging.info("Start accumulation")

    def save_and_update_accumulation_parameters_in_parachute(self):
        logging.warning(
            "self.accumulation_manager.ratio_accumulated_images %.4f"
            % self.accumulation_manager.ratio_accumulated_images
        )
        self.video.ratio_accumulated_images = (
            self.accumulation_manager.ratio_accumulated_images
        )
        self.video.percentage_of_accumulated_images.append(
            self.video.ratio_accumulated_images
        )
        self.list_of_fragments.save(
            self.video.accumulation_folder / "list_of_fragments.json"
        )

    def save_after_second_accumulation(self):
        logging.info("Saving second accumulation parameters")
        # Save accumulation parameters
        self.save_and_update_accumulation_parameters_in_parachute()

        # Choose best accumulation
        self.video.accumulation_trial = int(
            np.argmax(self.video.percentage_of_accumulated_images)
        )

        # Update ratio of accumulated images and  accumulation folder
        self.video.ratio_accumulated_images = (
            self.video.percentage_of_accumulated_images[self.video.accumulation_trial]
        )
        self.video.create_accumulation_folder()

        # Load light list of fragments with identities of the best accumulation
        self.list_of_fragments = ListOfFragments.load(
            self.video.auto_accumulation_folder / "list_of_fragments.json"
        )

        # Save objects
        self.list_of_fragments.save(self.video.fragments_path)
        self.list_of_global_fragments.save(self.video.global_fragments_path)

        # set restoring folder
        logging.info("Restoring networks to best second accumulation")
        self.accumulation_network_params.restore_folder = self.video.accumulation_folder

        # TODO: allow to train only the fully connected layers
        self.accumulation_network_params.scopes_layers_to_optimize = [
            "fully-connected1",
            "fully_connected_pre_softmax",
        ]
        logging.info("Initializing accumulation network")

        # Load pretrained network
        self.identification_model = LearnerClassification.load_model(
            self.accumulation_network_params
        )

        # # Re-initialize fully-connected layers
        # self.identification_model.apply(fc_weights_reinit)

        logging.info("Sending model and criterion to GPU")
        cudnn.benchmark = True  # make it train faster
        self.identification_model = self.identification_model.to(get_device())

        self.video.save()


def ask_about_protocol3(protocol3_action: str, n_error_frames: int) -> None:
    """Raises a CustomError if protocol3_action is abort or aks and user answers abortion"""
    logging.info("Protocol 3 action: %s", protocol3_action)

    if protocol3_action == "abort":
        raise CustomError(
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
            (
                "Protocol 3 is a very time consuming algorithm and, in most cases, it"
                " can be avoided by redefining the segmentation parameters. As"
                " [red]there are %d frames with more blobs than animals[/red], we"
                " recommend you to abort the tracking session now and go back to the"
                " Segmentation app focusing on not having reflections, shades, etc."
                " detected as blobs. Check the following general recommendations:\n   "
                " - Define a region of interest to exclude undesired noise blobs\n    -"
                " Shrink the intensity (or background difference) thresholds\n    -"
                " Toggle the use of the background subtraction\n    - Shrink the blob's"
                " area thresholds"
            ),
            n_error_frames,
            extra={"markup": True},
        )
    else:
        logging.info(
            (
                "Protocol 3 is a very time consuming algorithm and, in most cases, it"
                " can be avoided by redefining the segmentation parameters. As"
                " [bold]there are NOT frames with more blobs than animals[/bold], the"
                " video is unlikely to have non-animal blobs. Even so, you can choose"
                " to abort the tracking session and redefine the segmentation"
                " parameters (specially shrinking the intensity (or background"
                " difference) thresholds) or to continue with Protocol 3."
            ),
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
        raise CustomError(
            "This is not an actual error: protocol 3 was going to start"
            " but PROTOCOL3_ACTION is set to 'ask' and used aborted."
        )
    return
