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

import math
import random
from collections import defaultdict
from typing import Tuple, List

from .environment import Environment
from .state_data import StateData
from .action_data import ActionData
from . import utils
from . import stats
from .model_parameters import ModelParams

import numpy as np
from gym import spaces

from slbo.envs import BaseModelBasedEnv


class SafePlaceEnv(Environment, BaseModelBasedEnv):
    def __init__(self):
        if utils.verbose_slbo:
            print('(SafePlaceEnv) The oracle is being utilized.')

        self.last_action = len(ActionData) - 1
        assert self.last_action >= 0

        self.reservations: defaultdict = None
        self.referenced_reservations: defaultdict = None
        self.last_hour: int = None
        self.last_minute: int = None
        self.initial_temp_in: float = None

        self.do_transition = self.do_real_transition if utils.expert else self.do_expert_transition

        if utils.expert:
            starting_seed = 11
            self.seeds = list(range(starting_seed, starting_seed + utils.n_stages))

        self.model_parameters = ModelParams(None, None, None, None)

        self.times_initialized = 0
        self.day = -1

        self.observation_space = spaces.Box(low=np.array([0, 0., 0.,
                                                          float(utils.reservations_generation_min_temp - 10),
                                                          float(utils.reservations_generation_min_temp - 10)]),
                                            high=np.array([50, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                                           float(utils.reservations_generation_max_temp + 10),
                                                           float(utils.reservations_generation_max_temp + 10)]))
        # self.action_space = spaces.Discrete(8)
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([float(self.last_action)]))
        self.reward_range = (0, 1)

        self.state: StateData = None

        if utils.verbose_slbo:
            print('(SafePlaceEnv) Environment created.')

    def step(self, action: float):
        if utils.step_stats:
            s_stats_time = stats.step_stats_start()

        action: int = self.action_float_to_int(action)
        action: ActionData = ActionData.get_action(action)
        next_state, rewards = self.do_transition(self.state, action)

        if utils.step_stats:
            stats.step_stats_record_data(self.state, action, rewards, s_stats_time, iteration_number=0, times_bn=0)

        self.state = next_state

        observation = next_state.gym
        reward = rewards[0]
        done = next_state.is_terminal
        info = {}
        return observation, reward, done, info

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        """
        Compute rewards and terminal status of multiple transitions
        """
        length = states.shape[0]
        rewards = []
        for i in range(length):
            state = StateData(numpy_array=states[i])
            action: int = self.action_float_to_int(actions[i])
            action: ActionData = ActionData.get_action(action)
            next_state = StateData(numpy_array=next_states[i])

            reward, _, _, _ = self.get_reward(state, action, next_state)
            rewards.append(reward)

        rewards = np.array(rewards)
        dones = np.zeros_like(rewards, dtype=bool)
        return rewards, dones

    def action_float_to_int(self, action_float: float) -> int:
        if action_float < 0:
            utils.outside_range()
            return 0
        if action_float > self.last_action:
            utils.outside_range()
            return self.last_action

        action = int(action_float)
        if action_float - action >= 0.5:
            action += 1
        return action

    # def reset(self):
    #     self.initialize_env()
    #     return self.state.gym

    def initialize_env(self, completely_randomize_initial_state=True):
        string, _ = utils.generate_reservations_file()
        self.reservations, self.referenced_reservations, self.initial_temp_in = \
            utils.update_reservations(string=string, random_initial_temp_in=True)
        if utils.expert:
            completely_randomize_initial_state = False
        self.state, self.last_hour, self.last_minute = utils.initial_state(reservations_=self.reservations,
                                                                           initial_temp_in_=self.initial_temp_in,
                                                                           ignore_utils_values=
                                                                           completely_randomize_initial_state)
        if utils.verbose_slbo:
            print('(SafePlaceEnv) New day generated.')

    def real_reset(self):
        self.times_initialized += 1

        if utils.step_stats:
            utils.initialize_all(model_parameters=self.model_parameters, environment=None)

        if utils.reproducibility or utils.expert:
            if self.day == utils.n_stages - 1:
                utils.warning('(SafePlaceEnv) Nothing more to load! At next reset an error will be raised. '
                              'Returning current state...')
                return self.state.gym
            if not self.day < utils.n_stages:
                utils.error('(SafePlaceEnv) Fatal error! Something is wrong in the reproducibility configuration.')

        if utils.reproducibility:
            self.day += 1
            reservations_filepath = utils.reproducibility_reservations[self.day]
            self.reservations, self.referenced_reservations, _ = \
                utils.update_reservations(reservations_filepath=reservations_filepath, random_initial_temp_in=False)
            self.initial_temp_in = utils.reproducibility_temp_in[self.day]

            self.state, self.last_hour, self.last_minute = utils.initial_state(reservations_=self.reservations,
                                                                               initial_temp_in_=self.initial_temp_in,
                                                                               ignore_utils_values=False)
            if utils.verbose:
                print('(SafePlaceEnv) Day %d loaded for next stage.' % (self.day + 1))
        else:
            if utils.expert:
                self.day += 1
                random.seed(self.seeds[self.day])

                if utils.verbose:
                    print('(SafePlaceEnv) Day %d loaded for next stage.' % (self.day + 1))

            self.initialize_env(completely_randomize_initial_state=True)

        return self.state.gym

    def virtual_reset(self):
        self.times_initialized += 1
        self.initialize_env(completely_randomize_initial_state=True)
        return self.state.gym

    def seed(self, seed: int = None):
        """
        This method is ignored by the superclass.
        """
        pass

    def verify(self, n=2000, eps=1e-4):
        pass

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def real_transition_model(self, s: StateData, a: ActionData) -> StateData:
        hour = s.hour
        minute = s.minute
        people = s.people
        co2 = s.co2
        voc = s.voc
        temp_in = s.temp_in
        temp_out = s.temp_out

        # get next_hour and next_minute
        next_hour, next_minute = self.next_time(hour, minute)

        # get next_people and next_temp_out
        next_people = self.reservations[next_hour][next_minute].people
        next_temp_out = self.reservations[next_hour][next_minute].temp_out

        # get next_co2
        next_co2, potential_variation = self.next_co2(co2, people, a)

        # get next_voc
        next_voc = self.next_voc(voc, potential_variation, people, a)

        # get next_temp_in
        next_temp_in = self.next_temp_in(temp_in, temp_out, people, a)

        # get next_s
        next_s = StateData(next_hour, next_minute, next_people, next_co2, next_voc, next_temp_in, next_temp_out)

        return next_s

    def expert_transition_model(self, s: StateData, a: ActionData) -> StateData:
        hour = s.hour
        minute = s.minute
        people = s.people
        co2 = s.co2
        voc = s.voc
        temp_in = s.temp_in
        temp_out = s.temp_out

        # get next_hour and next_minute
        next_hour, next_minute = self.next_time(hour, minute)

        # get next_people and next_temp_out
        next_people = self.reservations[next_hour][next_minute].people
        next_temp_out = self.reservations[next_hour][next_minute].temp_out

        # get next_co2
        next_co2 = self.expert_next_co2(co2, people, a)

        # get next_voc
        next_voc = self.expert_next_voc(voc, people, a)

        # get next_temp_in
        next_temp_in = self.expert_next_temp_in(temp_in, temp_out, a)

        # get next_s
        next_s = StateData(next_hour, next_minute, next_people, next_co2, next_voc, next_temp_in, next_temp_out)

        return next_s

    def simulate(self, s: StateData, a: ActionData) -> (StateData, float):
        """
        Simulator of the environment
            Input:
                - s: state at time t
                - a: action at time t
            Output:
                - next_s: state at time t + 1
                - reward: reward for 's x a -> next_s'
        """

        next_s = self.real_transition_model(s, a)

        # get reward
        reward, _, _, _ = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, reward

    def do_real_transition(self, s: StateData, a: ActionData) -> (StateData, Tuple[float, float, float, float]):
        """
        Real environment
            Input:
                - s: state at time t
                - a: action at time t
            Output:
                - next_s: state at time t + 1
                - rewards: reward for 's x a -> next_s' and components of the main reward
        """

        next_s = self.real_transition_model(s, a)

        # get rewards
        rewards = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, rewards

    def do_expert_transition(self, s: StateData, a: ActionData) -> (StateData, Tuple[float, float, float, float]):
        next_s = self.expert_transition_model(s, a)

        # get rewards
        rewards = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, rewards

    @staticmethod
    def get_prediction_error(s: StateData, a: ActionData) -> List[int]:
        return [0, 0, 0]

    @staticmethod
    def next_co2(co2, people, action):
        if action.is_stack_ventilation_compatible:
            if people > 0:
                # following Teleszewski and Gładyszewska-Fiedoruk 2018 paper
                gamma = people / utils.room_volume
                next_co2 = utils.B * gamma * utils.time_delta * action.k + co2
                potential_variation = (next_co2 - co2) / co2
            else:
                delta = co2 - utils.outdoor_co2
                if action.is_ventilation_active:
                    factor = (1 - action.ach / 200)
                    if delta > utils.low_co2_descent_delta:
                        # when 'window_ach' is 8 h^(-1), co2 decreases by 38.73% in an hour
                        # when 'vent_low_power_ach' is 9 h^(-1), co2 decreases by 42.45% in an hour
                        next_co2 = co2 * factor
                        # for 'next_co2' to get lower than 'utils.outdoor_co2' you would need an ach greater than 40
                        # (but 'action.ach' is always between 0.1 and 9.97), so there is no need to verify that.
                    else:
                        next_co2 = co2 - (1 / (20 * utils.low_co2_descent_delta)) * delta * delta
                        # again, this equation will never lower 'next_co2' so that it goes below 'utils.outdoor_co2'.

                    potential_variation = factor - 1
                else:
                    # co2 concentration reaches equilibrium
                    next_co2 = co2
                    potential_variation = 0
        else:
            if action.ach >= 15:
                # This calculation depends on the assumption 'utils.max_people' = 50
                new_ach = action.ach * 50 / (10 / 49 * people + 39.8)
            else:
                # This calculation depends on the assumption 'utils.max_people' = 50
                min_new_ach = 14.9  # with 50 people
                max_new_ach = 0.3666 * action.ach + 13.34499  # with 0 people
                m, q = utils.find_linear_equation(0, max_new_ach, 50, min_new_ach)
                new_ach = m * people + q

            factor = math.exp(-(new_ach - 15.2) / 35)
            next_co2 = co2 * factor

            potential_variation = factor - 1

            if next_co2 < utils.outdoor_co2:
                next_co2 = utils.outdoor_co2

        return next_co2, potential_variation

    @staticmethod
    def next_voc(voc, potential_variation, people, action):
        if action.is_sanitizer_active:
            voc_removed = utils.voc_removal_rate
        else:
            voc_removed = 0

        if people > 0:
            voc_produced = utils.voc_produced_per_person * people
        else:
            voc_produced = 0

        voc_delta = voc_produced - voc_removed
        # NOTE: we assume that the voc produced gets immediately mixed with air
        voc_delta_concentration = voc_delta / utils.room_volume

        next_voc = voc + voc_delta_concentration

        if potential_variation < 0:
            next_voc = next_voc * (1 + potential_variation)

        if next_voc < utils.outdoor_voc:
            next_voc = utils.outdoor_voc

        return next_voc

    @staticmethod
    def next_temp_in(temp_in, temp_out, people, action):
        people_increase = people * utils.time_delta / (utils.room_volume * 5)
        if action.is_sanitizer_active:
            sanitizer_increase = (utils.voc_removal_rate / 1000) * utils.time_delta / utils.room_volume
        else:
            sanitizer_increase = 0

        temp_increase = people_increase + sanitizer_increase

        if action.is_window_open:
            delta = temp_out - temp_in
            sign = math.copysign(1, delta)
            delta = abs(delta)

            if delta < 1.6:
                new_delta = 0.08 * delta * delta * utils.time_delta
            elif delta > 2.5:
                new_delta = 0.4 * utils.time_delta
            else:
                new_delta = (delta * utils.temp_m + utils.temp_q) / 5 * utils.time_delta

            new_delta = math.copysign(new_delta, sign)

            new_delta += (temp_increase / 4)

            next_temp_in = temp_in + new_delta
        else:
            next_temp_in = temp_in + temp_increase

        return next_temp_in

    @staticmethod
    def expert_next_co2(co2, people, action):
        if action.is_stack_ventilation_compatible:
            if people > 0:
                next_co2 = co2 + people * utils.time_delta / (action.ach * 150)
            else:
                if action.is_ventilation_active:
                    next_co2 = co2 - action.ach * 20000 / utils.room_volume
                else:
                    next_co2 = co2 - 100
        else:
            next_co2 = co2 - action.ach * 30000 / utils.room_volume

        if next_co2 < utils.outdoor_co2:
            next_co2 = utils.outdoor_co2

        return next_co2

    @staticmethod
    def expert_next_voc(voc, people, action):
        if action.is_sanitizer_active and action.is_ventilation_active:
            voc_removed = utils.voc_removal_rate * action.ach / 7 * 2
        elif not action.is_sanitizer_active and action.is_ventilation_active:
            voc_removed = utils.voc_removal_rate * action.ach / 7
        elif action.is_sanitizer_active and not action.is_ventilation_active:
            voc_removed = utils.voc_removal_rate
        else:
            voc_removed = 0

        if people > 0:
            voc_produced = 100 * people
        else:
            voc_produced = 0

        voc_delta = voc_produced - voc_removed
        voc_delta_concentration = voc_delta / utils.room_volume

        next_voc = voc + voc_delta_concentration

        if next_voc < utils.outdoor_voc:
            next_voc = utils.outdoor_voc

        return next_voc

    @staticmethod
    def expert_next_temp_in(temp_in, temp_out, action):
        if action.is_window_open:
            delta = temp_out - temp_in
            sign = math.copysign(1, delta)
            delta = abs(delta)

            new_delta = delta / 1.5

            new_delta = math.copysign(new_delta, sign)
            next_temp_in = temp_in + new_delta
        else:
            next_temp_in = temp_in

        return next_temp_in

    @staticmethod
    def get_reward(s: StateData, a: ActionData, next_s: StateData) -> Tuple[float, float, float, float]:
        """
        - air_quality (co2, voc)
        - comfort (temp, noise)
        - energy_consumption (energy)
        """

        people = s.people

        co2 = next_s.co2
        voc = next_s.voc
        temp_in = next_s.temp_in

        if people > 0:
            # air quality
            if co2 < utils.acceptable_co2:
                co2_reward = utils.co2_m * co2 + utils.co2_q
                if co2 < utils.outdoor_co2:
                    co2_reward = 1

            elif co2 < utils.ideal_max_co2:
                co2_reward = utils.co2_a * math.pow(co2, 2) + utils.co2_b * co2 + utils.co2_c
            else:
                co2_reward = 0

            if voc < utils.acceptable_voc:
                voc_reward = utils.voc_m * voc + utils.voc_q
                if voc < utils.outdoor_voc:
                    voc_reward = 1

            elif voc < utils.ideal_max_voc:
                voc_reward = utils.voc_a * math.pow(voc, 2) + utils.voc_b * voc + utils.voc_c
            else:
                voc_reward = 0

            air_quality_reward = ((co2_reward + voc_reward) / 2) * utils.air_quality_factor

            # comfort
            # devINFO: new update
            temp_reward = math.exp(-(((temp_in - 20) / 5) * ((temp_in - 20) / 5)))
            # temp_reward = math.exp(-(((temp_in - 20) / 2) * ((temp_in - 20) / 2)))

            noise_reward = a.noise_reward

            comfort_reward = ((3 * temp_reward + noise_reward) / 4) * utils.comfort_factor
        else:
            air_quality_reward = 1
            comfort_reward = 1

        # energy consumption
        energy_reward = a.energy_reward * utils.energy_factor

        reward = (air_quality_reward + comfort_reward + energy_reward) / \
                 (utils.air_quality_factor + utils.comfort_factor + utils.energy_factor)

        return reward, air_quality_reward, comfort_reward, energy_reward

    def next_time(self, hour, minute):
        minutes = self.reservations[hour].keys()

        # get next_hour and next_minute
        if max(minutes) == minute:
            index_of_hour = self.referenced_reservations['hours']['ki'][hour]
            index_of_next_hour = index_of_hour + 1

            if index_of_next_hour in self.referenced_reservations['hours']['ik']:
                next_hour = self.referenced_reservations['hours']['ik'][index_of_next_hour]
            else:
                utils.error('(simulate) Error! Could not find next hour')

            next_minute = self.referenced_reservations['minutes'][next_hour]['ik'][0]
        else:
            next_hour = hour

            index_of_minute = self.referenced_reservations['minutes'][hour]['ki'][minute]
            index_of_next_minute = index_of_minute + 1

            if index_of_next_minute in self.referenced_reservations['minutes'][hour]['ik']:
                next_minute = self.referenced_reservations['minutes'][hour]['ik'][index_of_next_minute]
            else:
                utils.error('(simulate) Error! Could not find next minute')

        return next_hour, next_minute
