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
from idtrackerai.network.learners.learners import Learner_Classification
from idtrackerai.network.utils.utils import fc_weights_reinit, weights_xavier_init
from idtrackerai.utils import conf, create_dir, json_object_hook

from .accumulation_manager import AccumulationManager
from .accumulator import perform_one_accumulation_step
from .assigner import assign_remaining_fragments
from .identity_transfer import identify_first_global_fragment_for_accumulation
from .network.network_params import NetworkParams
from .pre_trainer import pre_train_global_fragment


class TrackerAPI:
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
            self.knowledge_transfer_info_dict: dict = json.loads(
                kt_info_dict_path.read_text(), object_hook=json_object_hook
            )
        else:
            self.knowledge_transfer_info_dict = {}

        # Old requirements for restoring
        self.processes_to_restore = {}

        self.number_of_identities = None  # Number of identities
        self.accumulation_network_params: NetworkParams
        self.restoring_first_accumulation = False  # Flag restores first accumulation
        self.accumulation_step_finished = False  # Flag accumulation step finished

    def track_single_animal(self):
        logging.debug("Assigning identity 1 to all blobs")
        for bf in self.list_of_blobs.blobs_in_video:
            for blob in bf:
                blob.identity = 1
                blob.P2_vector = [1.0]

    def track_single_global_fragment_video(self):
        logging.info("TRACKING SINGLE GLOBAL FRAGMENT")

        def get_P2_vector(identity, number_of_animals):
            P2_vector = np.zeros(number_of_animals)
            P2_vector[identity - 1] = 1.0
            return P2_vector

        fragment_identifier_to_id = {}
        identity = 1
        for fragment in self.list_of_fragments.fragments:
            if fragment.is_an_individual:
                fragment_identifier_to_id[fragment.identifier] = identity
                identity += 1
            else:
                fragment_identifier_to_id[fragment.identifier] = None

        for bf in self.list_of_blobs.blobs_in_video:
            for b in bf:
                if b.is_an_individual:
                    b.identity = fragment_identifier_to_id[b.fragment_identifier]
                    b.P2_vector = get_P2_vector(
                        fragment_identifier_to_id[b.fragment_identifier],
                        self.video.number_of_animals,
                    )
        self.video.first_frame_first_global_fragment = [0]  # in case

    def track_with_identities(self) -> ListOfFragments:
        """In protocol 3, list_of_fragments is loaded from accumulation
        folders so the reference from outside tracker_API is lost.
        That's why list_of_fragments has to be returned"""
        self.video.tracking_timer.start()
        self._track_with_protocols_cascade()
        self.video.tracking_timer.finish()
        return self.list_of_fragments
        # track_with_cascade = True
        # if track_with_cascade:
        #     # This runs the protocol cascade and also the residual
        #     # identification, the impossible_jumps, the creation of
        #     # trajectories, the crossings interpolation, and the
        #     # creation of trajectories_wo_gaps
        #     # TODO: Factorize track_with_protocols_cascade so it only runs
        #     # up to residual identification
        #     self._track_with_protocols_cascade()
        # else:
        #     # TODO: Here is where new tracking methods should come
        #     # Call to tracking method

        #     # Call to postprocessing
        #     # TODO: Factorize postprocess_impossible_jumps
        #     # postprocess_impossible_jumps
        #     # create_trajectories
        #     # crossings_interpolation
        #     # create_trajectories_wo_gaps
        #     self.postprocess_impossible_jumps()
        #     raise NotImplementedError("New tracking methods are not allwoed")

    def _track_with_protocols_cascade(self):
        logging.info("******* Start tracking with protocol cascade ********")
        # Restoring
        if "protocols1_and_2" in self.processes_to_restore:
            delete = not self.processes_to_restore["protocols1_and_2"]
        else:
            delete = True

        # Create accumulation folder
        self.video.create_accumulation_folder(iteration_number=0, delete=delete)

        # Set number of animals params for identity transfer
        self.number_of_identities = (
            self.knowledge_transfer_info_dict["number_of_classes"]
            if self.video.identity_transfer
            else self.video.number_of_animals
        )

        self.init_accumulation_idCNN_params()

        # Restoring
        self.restoring_first_accumulation = False
        if (
            "post_processing" in self.processes_to_restore
            and self.processes_to_restore["post_processing"]
        ):
            raise NotImplementedError()
            # self.restore_trajectories()
            # self.restore_crossings_solved()
            # self.restore_trajectories_wo_gaps()

        if (
            "residual_identification" in self.processes_to_restore
            and self.processes_to_restore["residual_identification"]
        ):
            raise NotImplementedError()
            # if self.video.track_wo_identities:
            # TODO: bring restoring back to life
            # raise
            # self.restore_trajectories()

            # else:
            # TODO: bring restoring back to life
            # raise
            # logging.info("Restoring residual identification")
            # self.restore_identification()
            # self.create_trajectories()

        if (
            "protocol3_accumulation" in self.processes_to_restore
            and self.processes_to_restore["protocol3_accumulation"]
        ):
            raise NotImplementedError()
            # logging.info("Restoring second accumulation")
            # # self.restore_second_accumulation()
            # self.video._first_frame_first_global_fragment = (
            #     self.video._first_frame_first_global_fragment
            # )
            # logging.warning(
            #     "first_frame_first_global_fragment "
            #     + str(
            #         self.video.first_frame_first_global_fragment
            #     )
            # )
            # logging.info("Starting identification")
            #
            # self.create_trajectories()

        if (
            "protocol3_pretraining" in self.processes_to_restore
            and self.processes_to_restore["protocol3_pretraining"]
        ):
            # TODO: bring restoring back to life
            raise NotImplementedError()
            # logging.info("Restoring pretraining")
            # logging.info("Initialising pretraining network")
            # self.init_pretraining_net()
            # logging.info("Restoring pretraining")
            # self.accumulation_step_finished = True
            # self.restore_first_accumulation()
            # self.restore_pretraining()
            # self.accumulation_manager.ratio_accumulated_images =
            # self.video.percentage_of_accumulated_images[0]
            # self.video._first_frame_first_global_fragment = [
            #     self.video._first_frame_first_global_fragment[
            #         0
            #     ]
            # ]
            # self.video._percentage_of_accumulated_images = [
            #     self.video.percentage_of_accumulated_images[0]
            # ]
            # logging.info("Start accumulation parachute")
            #
            # self.accumulate()

        if (
            "protocols1_and_2" in self.processes_to_restore
            and self.processes_to_restore["protocols1_and_2"]
        ):
            # TODO: bring restoring back to life
            raise NotImplementedError()
            # logging.info("Restoring protocol 1 and 2")
            # self.restoring_first_accumulation = True
            # # self.restore_first_accumulation()
            # self.accumulation_manager.ratio_accumulated_images =
            # self.video.percentage_of_accumulated_images[0]
            # self.video._first_frame_first_global_fragment = [
            #     self.video._first_frame_first_global_fragment[
            #         0
            #     ]
            # ]
            # self.video._percentage_of_accumulated_images = [
            #     self.video.percentage_of_accumulated_images[0]
            # ]
            # self.accumulation_step_finished = True
            #
            # self.accumulate()

        if (
            "protocols1_and_2" not in self.processes_to_restore
            or not self.processes_to_restore["protocols1_and_2"]
        ):
            logging.info("Starting protocol cascade")
            self.protocol1()

    def init_accumulation_idCNN_params(self):
        self.accumulation_network_params = NetworkParams(
            number_of_classes=self.video.number_of_animals,
            architecture=conf.IDCNN_NETWORK_NAME,
            save_folder=self.video.accumulation_folder,
            knowledge_transfer_model_file=self.video.knowledge_transfer_folder,
            saveid="",
            model_name="identification_network",
            image_size=self.video.id_image_size,
            scopes_layers_to_optimize=conf.LAYERS_TO_OPTIMISE_PRETRAINING,
            loss="CE",
            print_freq=-1,
            use_gpu=True,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION},
            apply_mask=False,
            dataset="supervised",
            skip_eval=False,
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            plot_flag=False,
            return_store_objects=False,
            layers_to_optimize=conf.LAYERS_TO_OPTIMISE_ACCUMULATION,
            video_paths=self.video.video_paths,
        )
        # Save network params
        self.accumulation_network_params.save()

    def protocol1(self):
        self.video.protocol1_timer.start()

        # reset list of fragments and global fragments to fragmentation
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.reset(roll_back_to="fragmentation")

        # Initialize idCNN
        logging.info("Setting learner class")
        self.learner_class = Learner_Classification
        logging.info("Creating idCNN")
        if self.video.knowledge_transfer_folder:
            try:
                self.identification_model = self.learner_class.load_model(
                    self.accumulation_network_params, scope="knowledge_transfer"
                )
                logging.info("Tracking with knowledge transfer")
                if not self.video.identity_transfer:
                    logging.info("Reinitializing fully connected layers")
                    self.identification_model.apply(fc_weights_reinit)
                else:
                    logging.info(
                        "Identity transfer. Not reinitializing the fully connected layers."
                    )
            except RuntimeError:
                logging.error(
                    f"Could not load model {self.accumulation_network_params} to transfer"
                    " knowledge, following without knowledge nor identity transfer"
                )
                self.learner_class = Learner_Classification
                self.identification_model = self.learner_class.create_model(
                    self.accumulation_network_params
                )
                self.identification_model.apply(weights_xavier_init)
        else:
            self.identification_model = self.learner_class.create_model(
                self.accumulation_network_params
            )
            self.identification_model.apply(weights_xavier_init)

        # Set first global fragment to start accumulation.
        # The network is passed in case of identity transfer.
        logging.info("Setting first global fragment for accumulation")
        first_global_fragment = (
            self.list_of_global_fragments.set_first_global_fragment_for_accumulation(
                accumulation_trial=0
            )
        )

        self.video.first_frame_first_global_fragment.append(
            first_global_fragment.first_frame_of_the_core
            if first_global_fragment is not None
            else None
        )
        if first_global_fragment is not None:
            identify_first_global_fragment_for_accumulation(
                first_global_fragment,
                self.video,
                network_params=self.accumulation_network_params,
                identification_model=self.identification_model,
                knowledge_transfer_info_dict=self.knowledge_transfer_info_dict,
            )

        # Order global fragments by distance to the first global fragment for the accumulation
        self.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(
            self.video.first_frame_first_global_fragment, accumulation_trial=0
        )

        # Instantiate accumulation manager
        self.accumulation_manager = AccumulationManager(
            self.video.id_images_file_paths,
            self.video.number_of_animals,
            self.list_of_fragments,
            self.list_of_global_fragments,
        )

        # General counter for training epochs
        self.global_step = 0

        # Selecting the first global fragment is considered as
        # the 0 accumulation step
        self.accumulation_step_finished = True
        self.init_and_accumulate()

    def one_shot_accumulation(self):
        self.accumulation_step_finished = False
        assert self.identification_model is not None
        self.accumulation_manager.ratio_accumulated_images = (
            perform_one_accumulation_step(
                self.accumulation_manager,
                self.video,
                self.identification_model,
                self.learner_class,
                network_params=self.accumulation_network_params,
            )
        )
        self.accumulation_step_finished = True

    def accumulate(self):
        new_global_fragments_for_training = (
            self.accumulation_manager.new_global_fragments_for_training
        )
        if self.accumulation_step_finished and new_global_fragments_for_training:
            # Training and identification continues
            if (
                self.accumulation_manager.counter == 1
                and self.video.accumulation_trial == 0
            ):
                # first training finished
                self.video.protocol1_timer.finish()
                self.video.protocol2_timer.start()

            # Training and identification step
            self.one_shot_accumulation()
            # Re-enter the function for the next step of the accumulation
            self.accumulate()

        elif (
            not new_global_fragments_for_training
            and not self.video.protocol2_timer.finished
            and self.accumulation_manager.ratio_accumulated_images
            > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
        ):
            # Accumulation stop because protocol 1 is successful
            logging.info("--------------------> Protocol 1 successful")
            self.save_after_first_accumulation()
            if (
                "protocols1_and_2" not in self.processes_to_restore
                or not self.processes_to_restore["protocols1_and_2"]
            ):
                self.video.protocol1_timer.finish()

            self.identify()

        elif (
            not new_global_fragments_for_training
            and not self.video.protocol3_pretraining_timer.finished
        ):
            logging.info("--------------------> No more new global fragments")
            self.save_after_first_accumulation()

            if (
                self.accumulation_manager.ratio_accumulated_images
                >= conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            ):
                logging.info("--------------------> Protocol 2 successful")

                if (
                    "protocols1_and_2" not in self.processes_to_restore
                    or not self.processes_to_restore["protocols1_and_2"]
                ):
                    self.video.protocol2_timer.finish()

                self.identify()

            elif (
                self.accumulation_manager.ratio_accumulated_images
                < conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            ):
                logging.info(
                    "--------------------> Protocol 2 failed -> Start protocol 3"
                )
                # raise ValueError('Protocol 3')
                if (
                    "protocols1_and_2" not in self.processes_to_restore
                    or not self.processes_to_restore["protocols1_and_2"]
                ):
                    self.video.protocol1_timer.finish()
                    if self.video.protocol2_timer.started:
                        self.video.protocol2_timer.finish()
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
            logging.info(
                "--------------------> Accumulation Protocol 3 failed. Opening parachute ..."
            )
            if self.video.accumulation_trial == 0:
                self.video.protocol3_accumulation_timer.start()
            self.video.accumulation_trial += 1
            if (
                not new_global_fragments_for_training
                and self.video.accumulation_trial > 1
            ):
                self.save_and_update_accumulation_parameters_in_parachute()
            self.accumulation_parachute_init(self.video.accumulation_trial)

            self.init_and_accumulate()

        elif self.video.protocol3_pretraining_timer.finished and (
            self.accumulation_manager.ratio_accumulated_images
            >= conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            or self.video.accumulation_trial
            >= conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
        ):
            logging.info("Accumulation after protocol 3 has been successful")
            self.video.protocol3_accumulation_timer.finish()

            self.save_after_second_accumulation()
            logging.info("Start residual identification")
            self.identify()

        # Whether to re-enter the function for the next accumulation step
        if self.accumulation_manager.new_global_fragments_for_training:
            self.accumulate()

    def init_and_accumulate(self):
        """
        This is called in the first step of each accumulation trial
        :param do_accumulate:
        :return:
        """
        logging.info("Entering accumulation loop")
        self.video.init_accumulation_statistics_attributes()
        self.accumulate()

    def save_after_first_accumulation(self):
        """Set flags and save data"""
        logging.info("Saving first accumulation parameters")

        if not self.restoring_first_accumulation:
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
            self.pretrain_network_params.knowledge_transfer_model_file = (
                self.video.knowledge_transfer_folder
            )

        logging.info("Start pretraining")
        self.pretraining_step_finished = True
        self.pretraining_loop()

    def init_pretraining_variables(self):
        self.init_pretraining_net()
        self.pretraining_global_step = 0
        self.ratio_of_pretrained_images = 0

        # Initialize network
        self.learner_class = Learner_Classification
        logging.info("Creating model")
        if self.video.knowledge_transfer_folder:
            self.identification_model = self.learner_class.load_model(
                self.pretrain_network_params, scope="knowledge_transfer"
            )
            self.identification_model.apply(fc_weights_reinit)
        else:
            self.identification_model = self.learner_class.create_model(
                self.pretrain_network_params
            )
            self.identification_model.apply(weights_xavier_init)

    def init_pretraining_net(self):
        delete = (
            not self.processes_to_restore["protocol3_pretraining"]
            if "protocol3_pretraining" in self.processes_to_restore.keys()
            else True
        )
        create_dir(self.video.pretraining_folder, remove_existing=delete)

        self.pretrain_network_params = NetworkParams(
            number_of_classes=self.video.number_of_animals,
            architecture=conf.IDCNN_NETWORK_NAME,
            save_folder=self.video.pretraining_folder,
            saveid="",
            model_name="identification_network",
            image_size=self.video.id_image_size,
            scopes_layers_to_optimize=conf.LAYERS_TO_OPTIMISE_PRETRAINING,
            loss="CE",
            print_freq=-1,
            use_gpu=True,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION},
            apply_mask=False,
            dataset="supervised",
            skip_eval=False,
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            plot_flag=False,
            return_store_objects=False,
            layers_to_optimize=conf.LAYERS_TO_OPTIMISE_ACCUMULATION,
            video_paths=self.video.video_paths,
        )

    def pretraining_loop(self):
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.order_by_distance_travelled()
        self.one_shot_pretraining()
        self.continue_pretraining()

    def one_shot_pretraining(self):
        self.pretraining_step_finished = False
        self.pretraining_global_fragment = (
            self.list_of_global_fragments.global_fragments[self.pretraining_counter]
        )
        assert self.identification_model is not None
        (
            self.identification_model,
            self.ratio_of_pretrained_images,
            self.list_of_fragments,
            self.pretrained_model_path,
        ) = pre_train_global_fragment(
            self.video.number_of_animals,
            self.video.accumulation_step,
            self.identification_model,
            self.learner_class,
            self.pretrain_network_params,
            self.pretraining_global_fragment,
            self.list_of_fragments,
            self.pretraining_global_step,
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

    def accumulation_parachute_init(self, iteration_number):
        logging.debug("------------------------> accumulation_parachute_init")
        logging.info("Starting accumulation %i" % iteration_number)

        delete = (
            not self.processes_to_restore["protocol3_accumulation"]
            if "protocol3_accumulation" in self.processes_to_restore.keys()
            else True
        )

        self.video.create_accumulation_folder(
            iteration_number=iteration_number, delete=delete
        )
        self.video.accumulation_trial = iteration_number
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.reset(roll_back_to="fragmentation")

        # Initialize network
        if self.video.identity_transfer:
            logging.info("Load model for identity transfer")
            self.identification_model = self.learner_class.load_model(
                self.accumulation_network_params
            )
        else:
            self.identification_model = None

        logging.info("Setting first global fragment for accumulation")
        first_global_fragment = (
            self.list_of_global_fragments.set_first_global_fragment_for_accumulation(
                accumulation_trial=iteration_number - 1
            )
        )

        self.video.first_frame_first_global_fragment.append(
            first_global_fragment.first_frame_of_the_core
            if first_global_fragment is not None
            else None
        )
        if first_global_fragment is not None:
            identify_first_global_fragment_for_accumulation(
                first_global_fragment,
                self.video,
                network_params=self.accumulation_network_params,
                identification_model=self.identification_model,
                knowledge_transfer_info_dict=self.knowledge_transfer_info_dict,
            )

        # Sort global fragments by distance
        self.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(
            self.video.first_frame_first_global_fragment,
            accumulation_trial=iteration_number - 1,
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
        logging.info("Initialising accumulation network")

        # Load pretrained network
        self.identification_model = self.learner_class.load_model(
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
        self.global_step = 0

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
            self.video.accumulation_folder / "list_of_fragments.pickle"
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
            self.video.auto_accumulation_folder / "list_of_fragments.pickle"
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
        logging.info("Initialising accumulation network")

        # Load pretrained network
        self.identification_model = self.learner_class.load_model(
            self.accumulation_network_params
        )

        # # Re-initialize fully-connected layers
        # self.identification_model.apply(fc_weights_reinit)

        # Send model and criterion to GPU
        if self.accumulation_network_params.use_gpu:
            logging.info("Sending model and criterion to GPU")
            torch.cuda.set_device(0)
            cudnn.benchmark = True  # make it train faster
            self.identification_model = self.identification_model.cuda()

        self.video.save()

    """ Residual identification """

    def identify(self):
        self.video.identify_timer.start()
        assert self.identification_model is not None
        assign_remaining_fragments(
            self.list_of_fragments,
            self.identification_model,
            self.accumulation_network_params,
        )
        self.video.identify_timer.finish()
