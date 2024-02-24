# -*- coding: utf-8 -*-

"""
Online Model Adaptation in Monte Carlo Tree Search Planning

This file is part of free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

It is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the code.  If not, see <http://www.gnu.org/licenses/>.
"""


import os
import pickle
from time import time

import numpy as np
import tensorflow as tf
import execnet

import utils
from neural_network import NN, reset_weights, pypy_pickle
import execnet_pypy


def load_external_nn_data() -> (NN, str, int):
    """
    Loads the expert's model neural network weights. To achieve this, it uses the information provided in `utils.py`
    such us `utils.external_model_path` and `utils.already_completed_epochs`.

    Returns: i) the `NN` object, ii) a string that indicates the location of the `NNPyPy` object (i.e., the weights that
    will be used by `execnet_pypy.py`) in the file system and iii) an integer indicating the total number of epochs on
    which the neural network has already been trained for.
    """

    if utils.execnet and utils.execnet_interpreter_path is None:
        utils.warning('(nn) Warning! `utils.execnet_interpreter_path` is set to None. Unexpected behaviors could '
                      'happen.')

    # `model_path` and `dataset_path` are the only allowed variables to be assigned in this `NN` constructor while we
    # are inside this function
    if utils.external_model_path is None:
        utils.error('(nn) Assign a valid `external_model_path`.')
    nn = NN(model_path=utils.external_model_path, dataset_path=utils.external_dataset_path, shuffle=True)

    model_name = 'safeplace_simulated_nn_%s_%d_%s' % (str(nn.layers), utils.already_completed_epochs, nn.timestamp)
    initial_pickle_path = pypy_pickle(nn, model_name)

    return nn, initial_pickle_path, utils.already_completed_epochs


def train() -> None:
    """
    One of the two main functions used in our experiments (the other one is `mcts()` in `execnet_pypy.py`). It creates a
    PyPy process and an interface of communication (i.e., `channel` variable) thanks to `execnet` module that allows to
    exchange data between the processes. Notice that this file is executed by CPython (the standard interpreter), while
    `execnet_pypy.py` is executed by PyPy interpreter if `utils.execnet_interpreter_path` is set correctly.

    If `utils.skip_initial_training` is set to True, the function will load the pretrained NN through
    `load_external_nn_data()` function; otherwise, the new created NN will be pretrained for a number of epochs equal
    to `initial_epochs` before the main algorithm starts. Then, if `utils.initial_training_only` is set to False, the
    pretrained NN will be used in the following steps. In our experiments `utils.skip_initial_training` has always been
    set to True because the pretrained NN was already available.

    The function creates the PyPy process and a `channel` variable to immediately send the information about:
        - the number of times the main loops of this function and `execnet_pypy.py` one will run;
        - how many days (i.e., complete MCTS simulations) must pass before updating the neural network weights (in our
            experiments always set to 1);
        - the location of the datasets in the file system that will be augmented over time;
        - the timestamp indicating when the program started;
        - the location of the `NNPyPy` object which is serialized through `pickle` module.

    For the number of times set in the `batches` variable, the main loop executes the following operations:
        - waits until it receives the new transition data (i.e., the new observations from the real environment) from
            the PyPy process;
        - arranges and updates the dataset saved in memory. In particular, throughout the program execution, the dataset
            is always split in a fit (i.e., train) dataset and a validation dataset. The data ratio is decided by
            `utils.dataset_factor`. In our experiments it was set to 0.2 (i.e., 80% for the fit dataset and 20% for the
            validation one).
            As regards MCP_M variant, the expert's dataset is merged with the real transitions at every loop (the other
            variants do not consider the expert's dataset). The expert transitions that are considered close to the real
            ones (see `NN.update_from_support_datasets`) are removed from the dataset;
        - increases the batch size used for training if necessary (i.e., if a certain number of days have passed);
        - trains the neural network on the updated dataset and keeps the best weights found in this iteration of the
            loop;
        - extracts the weights of the neural network, saves them in a `NNPyPy` object and finally serializes it in the
            file system;
        - the location of the serialized object is sent to the PyPy process through the channel.
    """

    if isinstance(utils.tf_seed, int):
        tf.random.set_seed(utils.tf_seed)
        np.random.seed(utils.tf_seed)
        utils.warning('(nn) TensorFlow seed: %d' % utils.tf_seed)
    else:
        utils.warning('(nn) `tf_seed` is not an int value. The experiment will not be reproducible.')

    pypy3_path = utils.execnet_interpreter_path
    config = 'popen//python=' + pypy3_path
    dataset_path = 'datasets/expert_dataset/expert_dataset.csv'
    layers = (15, 30, 40)

    # Variables for pretrain
    already_completed_epochs = 0
    initial_epochs = 1000
    initial_batch_size = 64

    # Variables for training
    epochs = 500
    batch_size = 8
    # `simulations` * `batches` must be equal to the total number of reservations csv files you intend to use.
    # For instance, assuming you didn't modify datasets/generated_reservations_profiles folder, `utils.room` is set
    # to '01' and `utils.all_rooms` is set to False, then `simulations` * `batches` must be equal to 100 since the total
    # number of csv files inside `datasets/generated_reservations_profiles/01` folder is 100.
    simulations = 1  # number of days (i.e., complete MCTS simulations) between two NN updates.
                     # To imitate our experiments, leave it to 1.
    batches = 100    # total number of NN updates. Here `batches` has a different meaning from the usual one (see the
                     # above description of the function)

    if not utils.skip_initial_training:
        nn = NN(dataset_path=dataset_path,
                layers=layers,
                shuffle=True)
        timestamp = nn.timestamp

        model_name = 'safeplace_simulated_nn_%s_%d_%s' % (str(layers), initial_epochs, timestamp)
        model_path = utils.nn_folder + model_name + '.h5'

        # This version does not save the model
        if utils.tf_dataset:
            nn.fit_v2_tf(epochs=initial_epochs, batch_size=initial_batch_size, save_best=True)
        else:
            nn.fit_v2_numpy(epochs=initial_epochs, batch_size=initial_batch_size, save_best=True)

        initial_nn_pickle_path = pypy_pickle(nn, model_name)

        if utils.initial_training_only:
            if utils.verbose:
                print('(nn) Finished initial training. Exiting...')
            quit()
    else:
        initial_epochs = 0
        nn, initial_nn_pickle_path, already_completed_epochs = load_external_nn_data()
        timestamp = nn.timestamp
        layers = nn.layers

    datasets = (utils.observations_folder + 'fd_%s.csv' % timestamp,
                utils.observations_folder + 'vd_%s.csv' % timestamp)

    if utils.use_two_neural_networks:
        reset_weights(nn.model)

    gw = execnet.makegateway(config)
    channel = gw.remote_exec(source=execnet_pypy)
    # Sending `execnet_pypy.mcts` function args
    channel.send((batches, simulations, datasets, timestamp))

    channel.send(initial_nn_pickle_path)

    if utils.batch_stats:
        additional_stats = []

    if utils.verbose:
        print('(nn) Waiting for first data batch...', flush=True)
    for batch in range(batches):
        transition_data: list = channel.receive()
        if utils.verbose:
            print('(nn) Data received. Starting training n.%d...' % (batch + 1), flush=True)

        # Arranging data already scrambled
        fd_list: list = transition_data[0]
        vd_list: list = transition_data[1]

        def from_list_to_np(d_list: list) -> (np.ndarray, np.ndarray):
            string = ''.join(d_list)
            string = string[:-1]
            string = string.replace('\n', ';')
            d = np.matrix(string)
            x = d[:, :-3]
            y = d[:, -3:]
            x = np.array(x)
            y = np.array(y)

            return x, y

        fd_x, fd_y = from_list_to_np(fd_list)
        vd_x, vd_y = from_list_to_np(vd_list)

        if utils.tf_dataset:
            if not utils.optimize_dataset_between_batches:
                fd = tf.data.Dataset.from_tensor_slices((fd_x.tolist(), fd_y.tolist()))
                vd = tf.data.Dataset.from_tensor_slices((vd_x.tolist(), vd_y.tolist()))

                if batch == 0 and utils.discard_initial_dataset:
                    fd = nn.load_single_dataset(datasets[0])
                    vd = nn.load_single_dataset(datasets[1])

                    nn.set_fit_data(fd)
                    nn.set_validation_data(vd)
                else:
                    nn.set_fit_data(nn.fit_data.concatenate(fd))
                    nn.set_validation_data(nn.validation_data.concatenate(vd))
            else:
                total_rows_deleted = nn.update_from_support_datasets(fd_x, fd_y, vd_x, vd_y)
                if utils.batch_stats:
                    additional_stats.append(total_rows_deleted)
        else:
            if utils.optimize_dataset_between_batches:
                utils.error('(nn) Set `tf_dataset` to True. NumPy (i.e., legacy) mode is not supported yet.')

            if batch == 0 and utils.discard_initial_dataset:
                fd = nn.load_single_dataset(datasets[0])
                vd = nn.load_single_dataset(datasets[1])

                nn.set_fit_data(fd)
                nn.set_validation_data(vd)
            else:
                fd = (np.concatenate((nn.fit_data[0], fd_x)), np.concatenate((nn.fit_data[1], fd_y)))
                nn.set_fit_data(fd)

                vd = (np.concatenate((nn.validation_data[0], vd_x)), np.concatenate((nn.validation_data[1], vd_y)))
                nn.set_validation_data(vd)

        if utils.tf_dataset:
            fd_len = len(nn.fit_data)
            vd_len = len(nn.validation_data)
        else:
            fd_len = len(nn.fit_data[0])
            vd_len = len(nn.validation_data[0])

        if utils.verbose:
            print('(nn) Fit dataset length: %d' % fd_len)
            print('(nn) Validation dataset length: %d' % vd_len)

        # Increasing batch size dynamically
        # After 300 days
        if fd_len > 28800:
            batch_size = 64
        # After 200 days
        elif fd_len > 19200:
            batch_size = 32
        # After 100 days
        elif fd_len > 9600:
            batch_size = 16

        current_total_epochs = already_completed_epochs + initial_epochs + (batch + 1) * epochs
        model_name = 'safeplace_nn_%s_%d_%s' % (str(layers), current_total_epochs, timestamp)
        model_path = utils.nn_folder + model_name + '.h5'

        # This fit version saves the best weights found during training
        if utils.tf_dataset:
            nn.fit_v2_tf(epochs=epochs, batch_size=batch_size, save_best=True)
        else:
            nn.fit_v2_numpy(epochs=epochs, batch_size=batch_size, save_best=True)

        utils.warning('(INFO) Current memory usage: %.2f MB' % get_current_memory_usage())
        print(end='', flush=True)

        pickle_path = pypy_pickle(nn, model_name)

        if batch != batches - 1:
            channel.send(pickle_path)
            if utils.verbose:
                print('(nn) New weights location sent.', flush=True)

    if utils.batch_stats and utils.optimize_dataset_between_batches:
        folder = 'batch_stats/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder + 'safeplace_additional_statistics_' + timestamp + '.pickle'
        if utils.verbose:
            print('(nn) Saving additional statistics in the following file: ' + filepath, flush=True)
        with open(filepath, 'wb') as f:
            pickle.dump(additional_stats, f)

    if utils.verbose:
        print('(nn) All processes have finished their tasks! Exiting...', flush=True)


if __name__ == '__main__':
    def get_current_memory_usage() -> float:
        with open('/proc/self/status') as f:
            memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

        return int(memusage.strip()) / 1000.

    if not utils.execnet:
        utils.error('(execnet) Set `adaptability_execnet` to True.')

    if utils.compare_with_oracle and not utils.batch_stats:
        utils.error('(execnet) If `compare_with_oracle` is set to True, also `batch_stats` must be set to True.')

    if utils.optimize_dataset_between_batches and utils.external_dataset_path is None:
        utils.error('(execnet) If `optimize_dataset_between_batches` is set to True, provide `external_dataset_path`.')

    start = time()
    train()
    end = time()
    print('TOTAL time elapsed: ' + str(end - start))
