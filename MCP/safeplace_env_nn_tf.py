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

from neural_network import NN
from safeplace_env import SafePlaceEnv
from state_data import StateData
from action_data import ActionData


class SafePlaceEnvNNTF(SafePlaceEnv):
    """
    Subclass of `SafePlaceEnv`. The SafePlace environment that uses the neural network weights as the transition model
    through NumPy and TensorFlow libraries. Therefore, it is not possible to use PyPy interpreter.
    """
    def __init__(self, neural_network: NN) -> None:
        """
        Method called after the creation of the object. It prints some info to terminal and loads the neural network
        `NN` object. Then, it saves its weights.

        Args:
            neural_network: `NN` object representing the neural network.
        """
        if neural_network is None:
            utils.error('(SafePlaceEnvNNTF) provide `neural_network`.')

        self.nn = neural_network

        # Neural network's weights
        model = self.nn.model
        self.weights_length = len(model.layers)
        self.weights = [model.layers[i].get_weights() for i in range(1, self.weights_length, 1)]

        if utils.verbose:
            print('(SafePlaceEnvNNTF) Environment initialized.')

    def simulate(self, s: StateData, a: ActionData) -> (StateData, float):
        """
        Function used as interface for outside modules to obtain the next state and reward of the MDP transition.
        Exclusively used in MCTS simulation.

        Args:
            s: `StateData` object representing the current state.
            a: `ActionData` object representing the action.

        Returns: i) the next `StateData` object representing the next state in the MDP and ii) the resulting reward.
        """
        # get next_hour and next_minute
        next_hour, next_minute = SafePlaceEnv.next_time(s.hour, s.minute)

        # get next_people and next_temp_out
        next_people = utils.reservations[next_hour][next_minute].people
        next_temp_out = utils.reservations[next_hour][next_minute].temp_out

        window_open = 1 if a.is_window_open else 0
        sanitizer_active = 1 if a.is_sanitizer_active else 0

        x = np.array([s.people, s.co2, s.voc, s.temp_in, s.temp_out, window_open, a.ach, sanitizer_active])
        y = self.model_inference(self.nn, x)
        next_co2 = y[0]
        next_voc = y[1]
        next_temp_in = y[2]

        # get next_s
        next_s = StateData(next_hour, next_minute, next_people, next_co2, next_voc, next_temp_in, next_temp_out)

        # get reward
        reward, _, _, _ = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, reward

    def model_inference(self, nn: NN, x: np.ndarray) -> np.ndarray:
        """
        This function uses the neural network weights to predict the next state of the MDP. The results are used to
        create a new state node in the MCTS tree.

        Args:
            nn: `NN` object representing the neural network.
            x: input of the neural network.

        Returns: the output of the neural network as a NumPy array.
        """
        y = None
        for i in range(self.weights_length - 1):
            if nn.is_normalized and i == 0:
                y = (x - self.weights[i][0]) / np.sqrt(self.weights[i][1])
                continue
            y = np.matmul(y, self.weights[i][0]) + self.weights[i][1]
            if i == self.weights_length - 2:
                break
            y = np.maximum(y, 0)

        return y

    def get_prediction_error(self, s: StateData, a: ActionData) -> list[float, float, float]:
        """
        Compares the predictions of the neural network's model with the oracle's model ones and returns them as a list.

        Args:
            s: `StateData` object representing the current state.
            a: `ActionData` object representing the action.

        Returns: absolute values of prediction errors as a list.
        """
        real_next_s = SafePlaceEnv.transition_model(s, a)

        real_co2, real_voc, real_temp = real_next_s.co2, real_next_s.voc, real_next_s.temp_in

        other_next_s, _ = self.simulate(s, a)
        other_co2, other_voc, other_temp = other_next_s.co2, other_next_s.voc, other_next_s.temp_in

        errors = [abs(other_co2 - real_co2), abs(other_voc - real_voc), abs(other_temp - real_temp)]

        if utils.verbose:
            print('Prediction errors - absolute values - [`co2`, `voc`, `temp_in`]: ' + str(errors))

        return errors
