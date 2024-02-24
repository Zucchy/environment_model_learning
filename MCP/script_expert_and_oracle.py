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


import random
import os
import pickle
from datetime import datetime

import utils
import stats
from dataset_generation import DatasetGeneration
from state_data import StateData
from state_node import StateNode
from model_parameters import ModelParams
from safeplace_env import SafePlaceEnv
from execnet_pypy import uct


def expert_and_oracle_batch_stats(days) -> None:
    """
    This functions executes UCT algorithm using both expert and oracle environments. It also generates the related
    batch stats if `utils.batch_stats` is set to True. This allows to compare fairly these methods with all the other
    variants.

    Args:
        days: number of days to simulate. In our experiments the value is set to 100.
    """
    def oracle_uct(initial_state: StateData):
        new_env = SafePlaceEnv()
        old_env = utils.change_env(new_env)
        old_stats_path = utils.change_stats_path('oracle_safeplace/')
        old_stats_filename = utils.change_stats_filename(utils.stats_filename + '.oracle')

        root = StateNode(initial_state)
        cumulative_reward = 0

        while not root.state.is_terminal:
            if utils.step_stats:
                s_stats_time = stats.step_stats_start()

            for iteration_number in range(utils.model_param.iterations):
                root.simulate_from_state()

            best_an = max(root.actions.values(), key=lambda an: an.q())
            a = best_an.action
            s = root.state
            real_s, rewards = utils.env.do_transition(s, a)
            real_sn = StateNode(real_s)

            if utils.step_stats:
                stats.step_stats_record_data(s, a, rewards, s_stats_time, iteration_number, times_bn=0)

            cumulative_reward += rewards[0]

            # A new tree must be created in order to compare coherently with `uct` function
            root = real_sn

        utils.change_env(old_env)
        utils.change_stats_path(old_stats_path)
        utils.change_stats_filename(old_stats_filename)

        return cumulative_reward

    dg = DatasetGeneration()
    env = dg
    _timestamp = datetime.utcnow().strftime('%y%m%d_%H%M%S')

    exp_const = 10
    max_depth = 13
    rollout_moves = 0
    iterations_per_step = 10000

    # MCTS parameters
    mcts_param = ModelParams(
        exp_const=exp_const,
        max_depth=max_depth,
        rollout_moves=rollout_moves,
        iterations=iterations_per_step
    )

    _batches = days
    _simulations = 1

    # Reproducibility ---
    if utils.reproducibility:
        random.seed(utils.temperature_seed)

        csv_files = {}

        for root, dirs, files in os.walk(utils.generated_reservations_folder):
            if len(dirs) == 0:
                room = root.split('/')[-1]
                for idx, file in enumerate(files):
                    files[idx] = root + '/' + file
                csv_files[room] = sorted(files)

        csv_files = dict(sorted(csv_files.items()))

        if not utils.all_rooms:
            reproducibility_reservations = csv_files[utils.room][:days]
        else:
            reproducibility_reservations = []
            for key, item in csv_files.items():
                reproducibility_reservations += item

        utils.set_verbose(False)
        reproducibility_temp_in = {}
        for room, profile in csv_files.items():
            temps = []
            for reservations_filepath in profile:
                utils.update_reservations(reservations_filepath=reservations_filepath, random_initial_temp_in=True)
                temps.append(utils.initial_temp_in)
            reproducibility_temp_in[room] = temps
        utils.set_verbose(True)

        if not utils.all_rooms:
            reproducibility_temp_in = reproducibility_temp_in[utils.room][:days]
        else:
            reproducibility_temp_in_temp = []
            for key, item in reproducibility_temp_in.items():
                reproducibility_temp_in_temp += item
            reproducibility_temp_in = reproducibility_temp_in_temp

        if len(reproducibility_temp_in) != (_batches * _simulations):
            utils.error(
                '(mcts) Reproducibility error! Length of profile for room %s is not compatible with execnet '
                'preferences.' % utils.room)

        simulation_idx = 0
    # ---

    batches_stats = []

    for batch in range(_batches):
        cr_list = []
        oracle_cr_list = []
        co2_errors_list = []
        voc_errors_list = []
        temp_in_errors_list = []

        if utils.reproducibility:
            reservations_filepath = reproducibility_reservations[simulation_idx]
            utils.update_reservations(reservations_filepath=reservations_filepath)
            utils.set_initial_temp_in(reproducibility_temp_in[simulation_idx])
            simulation_idx += 1

        utils.initialize_all(mcts_param, env)

        initial_state = utils.initial_state()
        new_data, cr, co2_ae, voc_ae, temp_in_ae = uct(initial_state)

        if utils.compare_with_oracle:
            oracle_cr = oracle_uct(initial_state)
            oracle_cr_list.append(oracle_cr)

        if utils.batch_stats:
            cr_list.append(cr)
            co2_errors_list += co2_ae
            voc_errors_list += voc_ae
            temp_in_errors_list += temp_in_ae

        if utils.batch_stats:
            batch_stats = (cr_list, oracle_cr_list, co2_errors_list, voc_errors_list, temp_in_errors_list)
            batches_stats.append(batch_stats)

    if utils.batch_stats:
        folder = 'batch_stats/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder + 'safeplace_' + _timestamp + '.pickle'
        if utils.verbose:
            print('(mcts) Saving statistics in the following file: ' + filepath, flush=True)
        with open(filepath, 'wb') as f:
            pickle.dump(batches_stats, f)

    if utils.verbose:
        print('(mcts) All %d batches completed! Exiting...' % _batches, flush=True)


if __name__ == '__main__':
    # Change room modifying `utils.room` variable
    expert_and_oracle_batch_stats(days=100)
