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

import numpy as np
import tensorflow as tf

from neural_network import NN, pypy_pickle


def pretrain() -> None:
    """
    Executes the pretraining on the expert's dataset.
    """

    if isinstance(utils.tf_seed, int):
        tf.random.set_seed(utils.tf_seed)
        np.random.seed(utils.tf_seed)
        utils.warning('(nn) TensorFlow seed: %d' % utils.tf_seed)
    else:
        utils.warning('(nn) `tf_seed` is not an int value. The experiment will not be reproducible.')

    utils.set_tf_dataset(False)
    dataset_path = 'datasets/expert_dataset/expert_dataset.csv'
    layers = (15, 30, 40)

    # Variables for pretrain
    initial_epochs = 1000
    initial_batch_size = 64

    nn = NN(dataset_path=dataset_path,
            layers=layers,
            shuffle=True)
    timestamp = nn.timestamp

    model_name = 'safeplace_simulated_nn_%s_%d_%s' % (str(layers), initial_epochs, timestamp)
    model_path = utils.nn_folder + 'pretraining/' + model_name + '.h5'

    nn.fit(epochs=initial_epochs, batch_size=initial_batch_size, save_best=True, filepath=model_path)

    # Load best model
    nn.model = tf.keras.models.load_model(model_path)
    pypy_pickle(nn, model_name)

    print('(nn) Finished initial training. Exiting...')


if __name__ == '__main__':
    pretrain()
