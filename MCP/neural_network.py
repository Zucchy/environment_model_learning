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


import utils

import os
import gc
import pickle
from datetime import datetime
import time
from math import ceil
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

from matrix_pypy import NNPyPy


class NN:
    """
    Class responsible for all methods related to the neural network using TensorFlow, except the model inference.
    In addition, it can:
            - load the neural network's model from the file system or from memory if already present.
            - create a new model for the neural network.
            - load the training dataset as a `np.ndarray` or a `tf.data.Dataset` object.
            - normalize, shuffle and manipulate the dataset.
            - train the model with the two possible representation of the dataset.
    """
    def __init__(self, model: tf.keras.Model = None, model_path: str = None,
                 datasets: tuple = None, dataset_path: str = None,
                 layers: tuple = None, shuffle: bool = True) -> None:
        """
        Method called after the creation of the object. It prints some info during execution and prepares the dataset
        if given. If no model is provided, creates a new model.

        Args:
            model: `tf.keras.Model` object representing the model already loaded in memory. It cannot be ignored if
                `model_path` is set to None as well.
            model_path: filepath of the model. It cannot be ignored if `model` is set to None as well.
            datasets: tuple that consists of the fit dataset, validation dataset and test dataset. It can be ignored.
            dataset_path: filepath of the dataset. It can be ignored.
            layers: tuple that identifies the architecture of the hidden layers (i.e., number of nodes for each layer).
            shuffle: set to True if you want to shuffle the dataset at the beginning; otherwise, set to False.
                If `utils.tf_dataset` is set to True, the dataset will be shuffled anyway.
        """

        if utils.verbose:
            print('(NN) `shuffle` is set to %s.' % str(shuffle))

        self.timestamp = datetime.utcnow().strftime('%y%m%d_%H%M%S')

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.fit_metric = tf.keras.metrics.MeanSquaredError()
        self.val_metric = tf.keras.metrics.MeanSquaredError()

        # Support datasets utilized by MCP_M
        self.sfd_x = None
        self.sfd_y = None
        self.svd_x = None
        self.svd_y = None

        if model is not None and model_path is not None:
            utils.error('(NN) `model` and `model_path` can\'t be both assigned.')
        if datasets is not None and dataset_path is not None:
            utils.error('(NN) `datasets` and `dataset_path` can\'t be both assigned.')
        if model is None and model_path is None:
            if datasets is None and dataset_path is None:
                utils.error('(NN) when the model is not retrievable, `datasets` or `dataset_path` (exclusive or) is '
                            'needed.')

            if layers is None:
                self.layers = (15, 30, 40)
                if utils.verbose:
                    print('(NN) Automatically set neural network layers to %s.' % str(self.layers))
            else:
                if not len(layers) > 0:
                    utils.error('(NN) Don\'t provide an empty `layer` tuple.')
                self.layers = layers

            self.is_normalized = True
            self.model = self.create_model()

            # Legacy
            if not utils.tf_dataset:
                self.model.compile(loss='mean_squared_error', optimizer='adam')

            if datasets is None:
                self.fit_data, self.validation_data, self.test_data = \
                    self.load_data_and_normalize(dataset_path=dataset_path, shuffle=shuffle, timestamp=self.timestamp)
            else:
                self.fit_data, self.validation_data, self.test_data = datasets

            # Legacy
            if not utils.tf_dataset:
                norm = self.model.get_layer('normalization')
                norm.adapt(self.fit_data[0])
        else:
            if model is not None:
                if not isinstance(model, tf.keras.Model):
                    utils.error('(NN) `model` is not a keras model.')
                self.model = model

            if model_path is not None:
                self.model = tf.keras.models.load_model(model_path)

            if layers is not None:
                utils.warning('(NN) Why is `layers` assigned? The program obtains the layers information by itself.')

            self.is_normalized = False
            layers = []
            for idx, l in enumerate(self.model.layers):
                if idx == 0:
                    continue
                elif idx == 1 and l.name == 'normalization':
                    self.is_normalized = True
                    continue
                elif idx == len(self.model.layers) - 1:
                    continue
                else:
                    layers.append(l.output_shape[1])
            self.layers = tuple(layers)

            # Load a possible dataset
            if dataset_path is not None:
                self.fit_data, self.validation_data, self.test_data = \
                    self.load_data_and_normalize(dataset_path=dataset_path, shuffle=shuffle, timestamp=self.timestamp)
            elif datasets is not None:
                self.fit_data, self.validation_data, self.test_data = datasets
            else:
                NN.initialize_csv_datasets(self.timestamp)
                self.fit_data, self.validation_data, self.test_data = None, None, None

    @tf.function
    def train_step(self, x, y):
        """
        Executes the training of the neural network weights given a batch of fit data. This function has the decorator
        `@tf.function`. This speeds up the computation time. For further details about how `@tf.function` works,
        see also: https://www.tensorflow.org/guide/function

        Args:
            x: batch features.
            y: batch labels.

        Returns: loss metric of the current batch.
        """

        print('(train_step) Retracing...', flush=True)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.fit_metric.update_state(y, logits)

        return loss_value

    @tf.function
    def test_step(self, x, y):
        """
        Calculates the validation loss metric using a given batch of validation data. This function has the decorator
        `@tf.function`. This speeds up the computation time. For further details about how `@tf.function` works,
        see also: https://www.tensorflow.org/guide/function

        Args:
            x: batch features.
            y: batch labels.

        Returns: validation loss metric of the current batch.
        """

        print('(test_step) Retracing...', flush=True)
        val_logits = self.model(x, training=False)
        self.val_metric.update_state(y, val_logits)

    def fit_v2_tf(self, epochs: int, batch_size: int, fit_verbose: int = 2, save_best: bool = True) -> None:
        """
        Function that executes the training of the neural network weights for each epoch. It uses a `tf.data.Dataset`
        object that represents our dataset.

        Args:
            epochs: total number of epochs.
            batch_size: batch size of the dataset.
            fit_verbose: set to a value less than or equal to 2 if you only want info printed to the terminal.
            save_best: set to True if you want to override the weights obtained in the last epoch and save the best
                found throughout the training; otherwise, set to False.
        """

        min_val_loss = float('+inf')
        best_weights = None

        fit_dataset = self.fit_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                          deterministic=False).prefetch(20)

        validation_dataset = self.validation_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                                        deterministic=False).prefetch(20)

        for epoch in range(epochs):
            print('Epoch %d/%d - Batches: %d' % (epoch + 1, epochs, len(fit_dataset)))
            start = time.time()

            for step, (x_fit_batch, y_fit_batch) in enumerate(fit_dataset):
                loss_value = self.train_step(x_fit_batch, y_fit_batch)

            loss = self.fit_metric.result()
            loss = float(loss)
            self.fit_metric.reset_states()

            for x_val_batch, y_val_batch in validation_dataset:
                self.test_step(x_val_batch, y_val_batch)

            val_loss = self.val_metric.result()
            val_loss = float(val_loss)
            self.val_metric.reset_states()
            if fit_verbose <= 2:
                print('Loss: %.4f - Validation loss: %.4f - Time elapsed: %.2fs' % (loss, val_loss,
                                                                                    time.time() - start))
            if save_best and val_loss < min_val_loss:
                if fit_verbose <= 2:
                    print('Current validation loss is lower than %.4f. Saving model weights and overwriting previous '
                          'ones.' % min_val_loss)
                min_val_loss = val_loss
                best_weights = self.model.get_weights()

        if save_best:
            self.model.set_weights(best_weights)

    def fit_v2_numpy(self, epochs: int, batch_size: int, fit_verbose: int = 2, save_best: bool = True) -> None:
        """
        Function that executes the training of the neural network weights for each epoch. It uses a `np.ndarray` object
        that represents our dataset.

        Args:
            epochs: total number of epochs.
            batch_size: batch size of the dataset.
            fit_verbose: set to a value less than or equal to 2 if you only want info printed to the terminal.
            save_best: set to True if you want to override the weights obtained in the last epoch and save the best
                found throughout the training; otherwise, set to False.
        """

        min_val_loss = float('+inf')
        best_weights = None

        fit_batches = ceil(self.fit_data[0].shape[0] / batch_size)
        validation_batches = ceil(self.validation_data[0].shape[0] / batch_size)

        x_fit_batches = np.array_split(self.fit_data[0], fit_batches)
        y_fit_batches = np.array_split(self.fit_data[1], fit_batches)
        x_validation_batches = np.array_split(self.validation_data[0], validation_batches)
        y_validation_batches = np.array_split(self.validation_data[1], validation_batches)

        for epoch in range(epochs):
            print("Epoch %d/%d - Batches: %d" % (epoch + 1, epochs, fit_batches))
            start = time.time()

            for step, (x_fit_batch, y_fit_batch) in enumerate(zip(x_fit_batches, y_fit_batches)):
                x_fit_batch = tf.convert_to_tensor(x_fit_batch, dtype=tf.float32)
                y_fit_batch = tf.convert_to_tensor(y_fit_batch, dtype=tf.float32)
                loss_value = self.train_step(x_fit_batch, y_fit_batch)

            loss = self.fit_metric.result()
            loss = float(loss)
            self.fit_metric.reset_states()

            for x_val_batch, y_val_batch in zip(x_validation_batches, y_validation_batches):
                x_val_batch = tf.convert_to_tensor(x_val_batch, dtype=tf.float32)
                y_val_batch = tf.convert_to_tensor(y_val_batch, dtype=tf.float32)
                self.test_step(x_val_batch, y_val_batch)

            val_loss = self.val_metric.result()
            val_loss = float(val_loss)
            self.val_metric.reset_states()
            if fit_verbose <= 2:
                print('Loss: %.4f - Validation loss: %.4f - Time elapsed: %.2fs' % (loss, val_loss,
                                                                                    time.time() - start))
            if save_best and val_loss < min_val_loss:
                if fit_verbose <= 2:
                    print('Current validation loss is lower than %.4f. Saving and overwriting best model weights.' %
                          min_val_loss)
                min_val_loss = val_loss
                best_weights = self.model.get_weights()

        if save_best:
            self.model.set_weights(best_weights)

    def create_model(self) -> tf.keras.Model:
        """
        Creates the Keras model from the hidden layer architecture (i.e., `self.layers`) given during the creation of
        the `NN` object.

        Returns: the created Keras model.
        """

        input = tf.keras.Input(shape=8, name='input')
        h = input
        h = tf.keras.layers.experimental.preprocessing.Normalization()(h)

        for i in range(len(self.layers)):
            h = tf.keras.layers.Dense(self.layers[i], activation='relu', name='hidden_' + str(i))(h)

        y = tf.keras.layers.Dense(3, activation='linear', name='output')(h)
        model = tf.keras.Model(inputs=input, outputs=y)

        return model

    def load_data_and_normalize(self, dataset_path: str, shuffle: bool, timestamp: str) -> \
            (Union[tf.data.Dataset, np.ndarray, None], Union[tf.data.Dataset, np.ndarray, None], None):
        """
        Loads and normalizes the dataset. It chooses between NumPy or TensorFlow dataset depending on
        `utils.tf_dataset`.

        Args:
            dataset_path: location of the dataset (it must be in csv format) in the file system.
            shuffle: set to True if you want to shuffle the dataset at the beginning; otherwise, set to False.
                If `utils.tf_dataset` is set to True, the dataset will be shuffled anyway.
            timestamp: string representing the time of creation of `NN` object.

        Returns: i) the fit dataset, ii) the validation dataset and iii) None (since the test dataset is not used during
        training). If `utils.optimize_dataset_between_batches` is set to True, returns (None, None, None).
        """

        fd = pd.read_csv(dataset_path)

        if shuffle:
            fd = fd.sample(frac=1)

        rows_to_remove = int(fd.shape[0] * utils.dataset_factor)

        fd, vd = fd.drop(fd.head(rows_to_remove).index), fd.head(rows_to_remove)
        fd, vd = fd.reset_index(drop=True), vd.reset_index(drop=True)

        if utils.execnet:
            if not os.path.exists(utils.observations_folder):
                os.makedirs(utils.observations_folder)

            if utils.discard_initial_dataset:
                header = fd.drop(fd.index.to_list()[:])
                header.to_csv(utils.observations_folder + 'fd_%s.csv' % timestamp, index=False)
                header.to_csv(utils.observations_folder + 'vd_%s.csv' % timestamp, index=False)
            else:
                fd.to_csv(utils.observations_folder + 'fd_%s.csv' % timestamp, index=False)
                vd.to_csv(utils.observations_folder + 'vd_%s.csv' % timestamp, index=False)

        fd_x = fd.copy()
        fd_y = fd_x[['next_co2', 'next_voc', 'next_temp_in']].copy()
        fd_x = fd_x.drop(['next_co2', 'next_voc', 'next_temp_in'], axis=1)

        vd_x = vd.copy()
        vd_y = vd_x[['next_co2', 'next_voc', 'next_temp_in']].copy()
        vd_x = vd_x.drop(['next_co2', 'next_voc', 'next_temp_in'], axis=1)

        if utils.optimize_dataset_between_batches:
            if not utils.tf_dataset:
                utils.error('(NN) Set `tf_dataset` to True. NumPy (i.e., legacy) mode is not supported yet.')

            self.sfd_x, self.sfd_y, self.svd_x, self.svd_y = NN.add_expert_column(fd_x, fd_y, vd_x, vd_y, value=1)

            if utils.execnet:
                # The support datasets have just been initialized, so there is no need to return other datasets
                return None, None, None

        if utils.tf_dataset:
            fit_data_x = tf.data.Dataset.from_tensor_slices(fd_x.values.tolist()).batch(50000)
            norm = self.model.get_layer('normalization')
            norm.adapt(fit_data_x)

            if not utils.execnet and utils.optimize_dataset_between_batches:
                return None, None, None

            fit_data = tf.data.Dataset.from_tensor_slices((fd_x.values.tolist(), fd_y.values.tolist()))
            validation_data = tf.data.Dataset.from_tensor_slices((vd_x.values.tolist(), vd_y.values.tolist()))

            fit_data = fit_data.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
            validation_data = validation_data.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
        else:
            # Legacy
            fit_data = (fd_x.to_numpy(), fd_y.to_numpy())
            validation_data = (vd_x.to_numpy(), vd_y.to_numpy())

        return fit_data, validation_data, None

    @staticmethod
    def add_expert_column(fd_x: pd.DataFrame, fd_y: pd.DataFrame, vd_x: pd.DataFrame, vd_y: pd.DataFrame, value: int) \
            -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        This function creates the support datasets utilized by MCP_M variant and adds a column to distinguish between
        expert and real transitions. Only `fd_x` and `vd_x` datasets get edited.

        Args:
            fd_x: original fit dataset containing the features.
            fd_y: original fit dataset containing the labels.
            vd_x: original validation dataset containing the features.
            vd_y: original validation dataset containing the labels.
            value: integer assigned to expert transitions (at the beginning all rows of the dataset are expert
                transitions). It is advised to set this parameter to 1; otherwise, unexpected behaviour could occur.

        Returns: i) support fit dataset containing the features, ii) support fit dataset containing the labels,
        iii) support validation dataset containing the features, iv) support validation dataset containing the labels.
        """

        fd_expert = [value for _ in range(len(fd_x))]
        fd_x['expert'] = fd_expert

        if value == 1:
            vd_expert = fd_expert[:len(vd_x)]
            vd_x['expert'] = vd_expert

        return fd_x, fd_y, vd_x, vd_y

    def update_from_support_datasets(self, new_fd_x, new_fd_y, new_vd_x, new_vd_y) -> int:
        """
        This function and the other ones here included are exclusively used by MCP_M variant. The objective of this
        method is updating the support datasets with the real transitions that are provided at every completion of an
        iteration of PyPy process main loop. To achieve this goal, this method removes the expert rows of both fit and
        validation datasets that make the function `utils.states_are_close` return a True value at least once when
        compared with all the provided real transitions.

        Args:
            new_fd_x: fit dataset containing the features of the new real transitions.
            new_fd_y: fit dataset containing the labels of the new real transitions.
            new_vd_x: validation dataset containing the features of the new real transitions.
            new_vd_y: validation dataset containing the labels of the new real transitions.

        Returns: a statistic representing the total number of rows removed from the datasets
        """

        def update_dataframe(sd_x: pd.DataFrame, sd_y: pd.DataFrame, new_d_x: np.ndarray, new_d_y: np.ndarray,
                             new_other_d_x: np.ndarray):
            sd_x_people = sd_x['people']
            sd_x_co2 = sd_x['co2']
            sd_x_voc = sd_x['voc']
            sd_x_temp_in = sd_x['temp_in']
            sd_x_temp_out = sd_x['temp_out']
            sd_x_window_open = sd_x['window_open']
            sd_x_ach = sd_x['ach']
            sd_x_sanitizer_active = sd_x['sanitizer_active']
            sd_x_expert = sd_x['expert']

            d_action_zip = zip(sd_x_window_open, sd_x_ach, sd_x_sanitizer_active)
            d_state_zip = zip(sd_x_people, sd_x_co2, sd_x_voc, sd_x_temp_in, sd_x_temp_out)

            total_rows_deleted = 0

            def delete_dataframe_rows(_df_idx, _action, _state, _expert, _sd_x, _sd_y, _d_x):
                _found = False
                _rows_deleted = 0
                for np_idx, row in enumerate(_d_x):
                    new_state = (row[0], row[1], row[2], row[3], row[4])
                    new_action = (row[5], row[6], row[7])

                    if _expert == 1 and _action == new_action:
                        if utils.states_are_close(new_state, _state):
                            try:
                                _sd_x = _sd_x.drop(df_idx)
                                _sd_y = _sd_y.drop(df_idx)
                                _found = True
                                _rows_deleted += 1
                            except KeyError:
                                pass
                            break

                return _sd_x, _sd_y, _found, _rows_deleted

            for df_idx, action, state, expert in zip(sd_x.index, d_action_zip, d_state_zip, sd_x_expert):
                sd_x, sd_y, found, rows_deleted = delete_dataframe_rows(df_idx, action, state, expert, sd_x, sd_y,
                                                                        new_d_x)

                other_rows_deleted = 0
                if not found:
                    sd_x, sd_y, _, other_rows_deleted = delete_dataframe_rows(df_idx, action, state, expert, sd_x, sd_y,
                                                                              new_other_d_x)
                total_rows_deleted += rows_deleted + other_rows_deleted

            # insert new data
            new_d_x = pd.DataFrame(new_d_x, columns=sd_x.columns[:-1])
            new_d_y = pd.DataFrame(new_d_y, columns=sd_y.columns)
            new_d_x, _, _, _ = NN.add_expert_column(new_d_x, None, None, None, value=0)

            sd_x = pd.concat([sd_x, new_d_x])
            sd_y = pd.concat([sd_y, new_d_y])
            sd_x = sd_x.reset_index(drop=True)
            sd_y = sd_y.reset_index(drop=True)

            return sd_x, sd_y, total_rows_deleted

        # step 1: delete expert data inside support datasets following a given criterion
        # (i.e., `utils.states_are_close()`)
        # Update fit dataset
        self.sfd_x, self.sfd_y, fd_rows_deleted = update_dataframe(self.sfd_x, self.sfd_y, new_fd_x, new_fd_y, new_vd_x)

        # Update validation dataset
        self.svd_x, self.svd_y, vd_rows_deleted = update_dataframe(self.svd_x, self.svd_y, new_vd_x, new_vd_y, new_fd_x)

        # Drop `expert` column and convert datasets to lists
        fd_x = self.sfd_x.drop('expert', axis=1).values.tolist()
        vd_x = self.svd_x.drop('expert', axis=1).values.tolist()
        fd_y = self.sfd_y.values.tolist()
        vd_y = self.svd_y.values.tolist()

        # step 2: create `tf.data.Dataset` from updated dataset
        fit_data = tf.data.Dataset.from_tensor_slices((fd_x, fd_y))
        validation_data = tf.data.Dataset.from_tensor_slices((vd_x, vd_y))

        fit_data = fit_data.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
        validation_data = validation_data.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)

        # step 3: make the updated datasets available for training the neural network
        self.fit_data = fit_data
        self.validation_data = validation_data

        print('(optimize_dataset) Total rows deleted: %d' % (fd_rows_deleted + vd_rows_deleted))

        return fd_rows_deleted + vd_rows_deleted

    @staticmethod
    def load_single_dataset(dataset_path: str) -> Union[tf.data.Dataset, np.ndarray]:
        """
        Loads the dataset. It chooses between NumPy or TensorFlow dataset depending on `utils.tf_dataset`.
        If `utils.tf_dataset` is set to True, it shuffles the dataset.

        Args:
            dataset_path: location of the dataset (it must be in csv format) in the file system.

        Returns: the single dataset (i.e., it doesn't get split).
        """

        d = pd.read_csv(dataset_path)

        d_x = d.copy()
        d_y = d_x[['next_co2', 'next_voc', 'next_temp_in']].copy()
        d_x = d_x.drop(['next_co2', 'next_voc', 'next_temp_in'], axis=1)

        if utils.tf_dataset:
            dataset = tf.data.Dataset.from_tensor_slices((d_x.values.tolist(), d_y.values.tolist()))
            dataset = dataset.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
        else:
            # Legacy
            dataset = (d_x.to_numpy(), d_y.to_numpy())

        return dataset

    @staticmethod
    def initialize_csv_datasets(timestamp: str) -> None:
        """
        Creates two empty csv files (one for the fit dataset and the other for the validation dataset) that contain only
        the header and then save them in the file system.

        Args:
            timestamp: string representing the time of creation of `NN` object.
        """

        header = utils.dataset_header

        if not os.path.exists(utils.observations_folder):
            os.makedirs(utils.observations_folder)

        with open(utils.observations_folder + 'fd_%s.csv' % timestamp, 'w') as f:
            f.write(header)
        with open(utils.observations_folder + 'vd_%s.csv' % timestamp, 'w') as f:
            f.write(header)

    def set_fit_data(self, fd: Union[tf.data.Dataset, np.ndarray]) -> None:
        """
        Replaces the old fit dataset with the one provided.

        Args:
            fd: new fit dataset.
        """
        self.fit_data = fd

    def set_validation_data(self, vd: Union[tf.data.Dataset, np.ndarray]) -> None:
        """
        Replaces the old validation dataset with the one provided.

        Args:
            vd: new validation dataset.
        """
        self.validation_data = vd

    def set_test_data(self, td: Union[tf.data.Dataset, np.ndarray]) -> None:
        """
        Replaces the old test dataset with the one provided.

        Args:
            td: new test dataset.
        """
        self.test_data = td

    def fit(self, epochs: int, batch_size: int, fit_verbose: int = 2, save_best: bool = True, filepath: str = None):
        """
        Finds the best weights for the NN. This function is exclusively used for pretraining the NN in
        `script_expert_pretraining.py` and wraps `tf.keras.Model.fit` function. It is advised to use `execnet_tf.py`
        with `utils.skip_initial_training` set to False instead for a more up-to-date method.
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit for more detail about the wrapped Keras
        function.

        Args:
            epochs: total number of epochs.
            batch_size: batch size of the dataset.
            fit_verbose: degree of verbosity of `tf.keras.Model.fit` function.
            save_best: set to True if you want to override the weights obtained in the last epoch and save the best
                found throughout the training; otherwise, set to False.
            filepath: the location in the file system where the NN will be saved.

        Returns: an `History` object of Keras library containing all the statistics of training.
        """
        if self.fit_data is None:
            utils.error('(NN) Provide `fit_data`.')

        if save_best and filepath is None:
            utils.error('(NN) If `save_best` is set to True, you need to provide a valid `filepath`.')

        callbacks = []

        class ClearSessionCallBack(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                tf.keras.backend.clear_session()

        clear_session_callback = ClearSessionCallBack()
        callbacks.append(clear_session_callback)

        if save_best:
            verbose = 1 if utils.verbose else 0
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                                    monitor='val_loss',
                                                                    verbose=verbose,
                                                                    save_best_only=True)
            callbacks.append(save_best_callback)

        return self.model.fit(x=self.fit_data[0],
                              y=self.fit_data[1],
                              validation_data=self.validation_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=fit_verbose,
                              callbacks=callbacks)

    def save_model(self, dest: str) -> None:
        """
        Saves the Keras model of the `NN` object in the file system.

        Args:
            dest: location of the file system where the model will be saved.
        """
        self.model.save(dest)


def reset_weights(model: tf.keras.Model) -> None:
    """
    This function executes the reinitialization of the weights of a NN.

    Args:
        model: `tf.keras.Model` object representing the model of the NN.
    """

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))


def pypy_pickle(nn: NN, model_name: str) -> str:
    """
    This functions extracts the weights of the Keras model and saves them in a `NNPyPy` object. Then, it saves this
    object in the file system through `pickle` module. Thanks to `pickle`, the `NNPyPy` object can be retrieved by the
    part of the code which is executed with PyPy interpreter.

    Args:
        nn: `NN` object representing the neural network.
        model_name: name of the file that will be created.

    Returns: location in the file system where the serialized `NNPyPy` object has been saved.
    """
    model = nn.model

    length = len(model.layers)
    weights = [model.layers[i].get_weights() for i in range(1, length, 1)]

    w_py = []
    for i in range(length - 1):
        w_py.append((weights[i][0].tolist(), weights[i][1].tolist()))

    nn = NNPyPy(w_py, nn.is_normalized)
    folder = utils.nn_folder + 'pickle/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = folder + model_name + '.pickle'
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)

    return filepath
