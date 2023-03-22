import json
import logging
from pathlib import Path

import numpy as np

from idtrackerai.network import LearnerClassification
from idtrackerai.tracker.network.network_params import NetworkParams

# import matplotlib.pyplot as plt
# import shutil
# from idtrackerai.video import Video as v
# from images import extract_images_and_labels, extract_images_and_labels_from_list

# from idtrackerai.network.identification_model.network_params\
#     import NetworkParams
# from idtrackerai.network.identification_model.id_CNN import ConvNetwork
# from idtrackerai.network.identification_model.get_data import split_data_train_and_validation
# from idtrackerai.network.identification_model.store_accuracy_and_loss import Store_Accuracy_and_Loss
# from idtrackerai.network.identification_model.stop_training_criteria import Stop_Training
# from idtrackerai.network.identification_model.epoch_runner import EpochRunner


def load_identification_model(model_folder: Path):
    params_path = model_folder / "model_params.json"
    if params_path.is_file():
        with open(params_path, "rb") as file:
            params: dict = json.load(file)
    elif params_path.with_suffix(".npy").is_file():
        params: dict = np.load(
            model_folder / "model_params.npy", allow_pickle=True
        ).item()
    else:
        raise FileNotFoundError(params_path)

    identification_network_params = NetworkParams(
        schedule=params["schedule"],
        number_of_classes=params["number_of_classes"],
        architecture="idCNN",
        restore_folder=model_folder,
        model_name=params["model_name"],
        dataset=params["dataset"],
        saveid=params["saveid"],
        image_size=params["image_size"],
        use_gpu=True,
    )

    # Initialize network
    logging.info("Setting learner class")
    learner_class = LearnerClassification
    logging.info("Creating model")
    identification_model = learner_class.load_model(identification_network_params)
    return identification_model, identification_network_params

    # def update_checkpoint_paths(current_model_folder, old_model_folder):
    #     for sub_folder in ['conv', 'softmax']:
    #         checkpoint_path = os.path.join(current_model_folder, sub_folder,
    #                                        'checkpoint')
    #         if os.path.isfile(checkpoint_path):
    #             v.update_tensorflow_checkpoints_file(checkpoint_path,
    #                                                  old_model_folder,
    #                                                  current_model_folder)
    #         else:
    #             print('No checkpoint found in %s '
    #                   % os.path.join(current_model_folder, sub_folder))
    #
    # def get_old_model_folder(current_model_folder):
    #     checkpoint_path = os.path.join(current_model_folder, 'conv',
    #                                    'checkpoint')
    #     with open(checkpoint_path) as f:
    #         content = f.readlines()
    #     return os.path.split(os.path.split(content[0].split('"')[1])[0])[0]
    #
    # old_model_folder = get_old_model_folder(model_folder)
    # if model_folder != old_model_folder:
    #     update_checkpoint_paths(model_folder, old_model_folder)
    # model_info = np.load(os.path.join(model_folder, 'info.npy')).item()
    # network_params = NetworkParams(model_info['number_of_animals'],
    #                                learning_rate=0.005,
    #                                keep_prob=1.0,
    #                                scopes_layers_to_optimize=None,
    #                                save_folder=model_folder,
    #                                restore_folder=model_folder,
    #                                image_size=model_info['input_image_size'])
    # net = ConvNetwork(network_params)
    # net.restore()
    # return net


# def train_model_from(identification_images_file_path):
#     print("Training model from {}".format(identification_images_file_path))
#     if isinstance(identification_images_file_path, str):
#         images, labels, info = extract_images_and_labels(
#             identification_images_file_path
#         )
#         save_folder = os.path.join(
#             os.path.dirname(os.path.dirname(identification_images_file_path)), "model"
#         )
#         if not os.path.isdir(save_folder):
#             os.mkdir(save_folder)
#         else:
#             shutil.rmtree(save_folder)
#             os.mkdir(save_folder)
#         np.save(os.path.join(save_folder, "id_images_info.npy"), info)
#         if len(labels) != 0:
#             network_params = NetworkParams(
#                 len(set(labels)),
#                 learning_rate=0.005,
#                 keep_prob=1.0,
#                 scopes_layers_to_optimize=None,
#                 save_folder=save_folder,
#                 image_size=images[0].shape + (1,),
#             )
#             net = ConvNetwork(network_params)
#             train_model(net, np.asarray(images), np.asarray(labels) - 1)
#             return save_folder
#         else:
#             raise ValueError("The list of labels is empty")

#     elif isinstance(identification_images_file_path, list):
#         images, labels, info = extract_images_and_labels_from_list(
#             identification_images_file_path
#         )
#         print(len(images), len(labels), set(labels))
#         save_folder = os.path.dirname(
#             os.path.dirname(os.path.dirname(identification_images_file_path[0]))
#         )
#         save_folder = os.path.join(save_folder, "model")
#         if not os.path.isdir(save_folder):
#             os.mkdir(save_folder)
#         else:
#             shutil.rmtree(save_folder)
#             os.mkdir(save_folder)
#         np.save(os.path.join(save_folder, "id_images_info.npy"), info)
#         if len(labels) != 0:
#             network_params = NetworkParams(
#                 len(set(labels)),
#                 learning_rate=0.005,
#                 keep_prob=1.0,
#                 scopes_layers_to_optimize=None,
#                 save_folder=save_folder,
#                 image_size=images[0].shape + (1,),
#             )
#             net = ConvNetwork(network_params)
#             train_model(net, np.asarray(images), np.asarray(labels))
#             return save_folder
#         else:
#             raise ValueError("The list of labels is empty")


# def train_model(net, images, labels, plot_flag=True):
#     print("Training...")
#     store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(
#         net, name="training", scope="training"
#     )
#     store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(
#         net, name="validation", scope="training"
#     )
#     if plot_flag:
#         plt.ion()
#         fig, ax_arr = plt.subplots(3)
#         fig.canvas.set_window_title("Accuracy and loss")
#         fig.subplots_adjust(
#             left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5
#         )
#     # Instantiate data set
#     training_dataset, validation_dataset = split_data_train_and_validation(
#         net.params.number_of_animals, images, labels
#     )
#     # Convert labels to one hot vectors
#     training_dataset.convert_labels_to_one_hot()
#     validation_dataset.convert_labels_to_one_hot()
#     # Reinitialize softmax and fully connected
#     net.reinitialize_softmax_and_fully_connected()
#     # Train network
#     # compute weights to be fed to the loss function (weighted cross entropy)
#     net.compute_loss_weights(training_dataset.labels)
#     trainer = EpochRunner(
#         training_dataset, starting_epoch=0, print_flag=True, batch_size=50
#     )
#     validator = EpochRunner(
#         validation_dataset, starting_epoch=0, print_flag=True, batch_size=50
#     )
#     # set criteria to stop the training
#     stop_training = Stop_Training(
#         net.params.number_of_animals,
#         check_for_loss_plateau=True,
#         first_accumulation_flag=True,
#     )
#     global_step = 0
#     global_step0 = global_step

#     while not stop_training(
#         store_training_accuracy_and_loss_data,
#         store_validation_accuracy_and_loss_data,
#         trainer._epochs_completed,
#     ):
#         # --- Training
#         feed_dict_train = trainer.run_epoch(
#             "Training", store_training_accuracy_and_loss_data, net.train
#         )
#         # --- Validation
#         feed_dict_val = validator.run_epoch(
#             "Validation", store_validation_accuracy_and_loss_data, net.validate
#         )
#         # update global step
#         net.session.run(
#             net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed)
#         )
#         # Update counter
#         trainer._epochs_completed += 1
#         validator._epochs_completed += 1

#     if np.isnan(store_training_accuracy_and_loss_data.loss[-1]) or np.isnan(
#         store_validation_accuracy_and_loss_data.loss[-1]
#     ):
#         raise ValueError("The model diverged")
#     else:
#         global_step += trainer.epochs_completed
#         print(
#             "loss values in validation: %s"
#             % str(store_validation_accuracy_and_loss_data.loss[global_step0:])
#         )
#         # plot if asked
#         if plot_flag:
#             store_training_accuracy_and_loss_data.plot(ax_arr, color="r")
#             store_validation_accuracy_and_loss_data.plot(ax_arr, color="b")
#         net.save()
