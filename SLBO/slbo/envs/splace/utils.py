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
import sys
import csv
import random
import math
import pickle
from collections import defaultdict
from datetime import datetime

from . import stats

from slbo.utils.flags import FLAGS
from lunzi.Logger import logger


# SLBO ---
seed = None

reproducibility = FLAGS.env.reproducibility
verbose_slbo = False
evaluate = False

n_outside_action_range = 0


def set_seed(value):
    global seed
    seed = value


def outside_range():
    global n_outside_action_range
    n_outside_action_range += 1


def reset_action_range_stat():
    global n_outside_action_range
    n_outside_action_range = 0
# ---


# Experimental settings ---
temperature_seed = 1243

room: str = FLAGS.env.room
days_per_room = 100
all_rooms: bool = FLAGS.env.all_rooms

n_stages: int = FLAGS.slbo.n_stages

expert: bool = FLAGS.env.expert

generated_reservations_folder = 'datasets/generated_reservations_profiles/'

if not all_rooms:
    logger.warning('SafePlace room is %s' % room)
else:
    logger.warning('SafePlace `all_rooms` mode activated')
logger.warning('SafePlace environment will run for %d days' % n_stages)
# ---


# General preferences ---
pypy = False
check_tree_integrity = False
verbose = True                # print main info on terminal
adaptability_nn_tf = False    # adaptability through neural network using TensorFlow
adaptability_nn_pypy = False  # adaptability through neural network with PyPy
execnet = True                # adaptability through neural network with execnet
step_stats = True             # save general statistics for every iteration/step on a csv file

reservation_profile_path = 'datasets/reservations_summer.csv'
# ---


# Execnet preferences ---
execnet_interpreter_path = '/home/edo/pypy3.9/bin/pypy3'
tf_dataset = True
skip_initial_training = True
initial_training_only = False
observations_folder: str = 'datasets/observations/'
discard_initial_dataset = True
polyak = False
polyak_tau = 0.5
use_two_neural_networks = False
optimize_dataset_between_batches = True

batch_stats = True
compare_with_oracle = False
save_reservations_history = True
use_reservations_history = False

# User input
reservations_history_path = 'batch_stats/foo.csv'
external_model_path = 'nn/pretrain/safeplace_simulated_nn_(15, 30, 40)_1000_220805_163956.h5'
external_dataset_path = 'datasets/expert_dataset/expert_dataset.csv'
already_completed_epochs: int = 1000
# ---


# Neural network settings --
nn_folder = 'nn/'
dataset_factor = 0.2  # percentage dedicated to validation dataset
dataset_header = 'people,co2,voc,temp_in,temp_out,window_open,ach,sanitizer_active,next_co2,next_voc,next_temp_in\n'
# ---


# `use_two_neural_networks` and `optimize_dataset_between_batches` variants settings ---
people_threshold = 10
co2_threshold = 300
voc_threshold = 300
temp_in_threshold = 5
temp_out_threshold = 5


def states_are_close(state: tuple, state_observed: tuple):
    s = state
    so = state_observed

    if so[4] - temp_out_threshold <= s[4] <= so[4] + temp_out_threshold:
        if so[3] - temp_in_threshold <= s[3] <= so[3] + temp_in_threshold:
            if so[0] - people_threshold <= s[0] <= so[0] + people_threshold:
                if so[2] - voc_threshold <= s[2] <= so[2] + voc_threshold:
                    if so[1] - co2_threshold <= s[1] <= so[1] + co2_threshold:
                        return True
    return False
# ---


# Environment settings ---
reservations_generation_first_hour = 8
reservations_generation_last_hour = 18
reservations_generation_min_temp = -5
reservations_generation_max_temp = 40

random_initial_temp_in_min = 17
random_initial_temp_in_max = 30
random_initial_temp_in_threshold = 5
random_initial_temp_out_max = 30

max_people = 50
time_delta = 5  # [min] do not change this variable unless you've already modified 'reservations_*.csv' file

B = 180                                          # [m^3*ppm/(person*min)] (Teleszewski and Gładyszewska-Fiedoruk 2018)
maximum_ach_stack_ventilation_compatible = 9.97  # do not change this constant
voc_produced_per_person_original = 6250          # [µg/(h*person)] (Tang et al. 2016)
voc_produced_per_person = voc_produced_per_person_original / 60 * time_delta  # [µg/('time_delta'min*person)]

voc_removal_rate = 10000   # [µg/'time_delta'min] voc removed by sanitizer per step

room_volume = 300       # [m^3]
initial_temp_in = 19.0  # [°C]
initial_ach = 0.1       # [h^(-1)]
initial_co2 = 500       # [ppm]
initial_voc = 50        # [µg/m^3]

acceptable_co2 = 1000
acceptable_voc = 600

outdoor_co2 = 400
low_co2_descent_delta = 100

outdoor_voc = 30

no_ventilation_ach = 0.1
window_ach = 8
vent_low_power_ach = 11
vent_medium_power_ach = 15
vent_high_power_ach = 21

# Reward constants
ideal_max_co2 = 2500
ideal_max_voc = 1500

max_reward_co2 = 1
min_reward_co2 = 0.7
max_reward_voc = 1
min_reward_voc = 0.7


def find_linear_equation(x1, y1, x2, y2):
    __m = (y1 - y2) / (x1 - x2)
    __q = y1 - __m * x1
    # 'y = %.5f * x + %.5f' % (m, q)
    return __m, __q


def find_parabola_with_discriminant_zero(x_min_reward, min_reward, x_zero_reward):
    _min = min_reward
    acceptable = x_min_reward  # beta
    ideal_max = x_zero_reward  # alpha

    __d = math.pow(ideal_max - acceptable, 2)

    __a = _min / __d
    __b = (-2) * ideal_max * _min / __d
    __c = math.pow(ideal_max, 2) * _min / __d

    return __a, __b, __c


temp_m, temp_q = find_linear_equation(1.6, 1.024, 2.5, 2)

co2_m, co2_q = find_linear_equation(outdoor_co2, max_reward_co2, acceptable_co2, min_reward_co2)
co2_a, co2_b, co2_c = find_parabola_with_discriminant_zero(acceptable_co2, min_reward_co2, ideal_max_co2)
voc_m, voc_q = find_linear_equation(outdoor_voc, max_reward_voc, acceptable_voc, min_reward_voc)
voc_a, voc_b, voc_c = find_parabola_with_discriminant_zero(acceptable_voc, min_reward_voc, ideal_max_voc)

# Reward factors
air_quality_factor = 1
comfort_factor = 1
energy_factor = 0.1
# ---

# Later imports for initializations
from slbo.envs.splace.state_data import StateData
from slbo.envs.splace.action_data import ActionData
from slbo.envs.splace.model_parameters import ModelParams
from slbo.envs.splace.environment import Environment

# These variables are initialized automatically, don't change them ---
last_hour: int
last_minute: int
line_count: int  # Number of steps of mcts simulation (i.e., lines in reservation profile string)

model_param: ModelParams
env: Environment

counter = 0
step_stats_counter = '000'

last_chosen_nn: bool
nn_hits: int = 0
nn_miss: int = 0

stats_path: str
stats_filename: str
timestamp: str

step_stats_folder = 'step_stats/'
image_folder = 'img/'

reservations: defaultdict
referenced_reservations: defaultdict
# ---


def set_step_stats(value):
    global step_stats
    step_stats = value


def set_time_delta(value):
    global time_delta
    time_delta = value


def set_initial_temp_in(new_value: float):
    global initial_temp_in
    initial_temp_in = new_value


def change_env(new_env: Environment) -> Environment:
    global env
    old_env = env
    env = new_env
    return old_env


def increase_step_stats_counter():
    global step_stats_counter
    n_digits = len(step_stats_counter)
    _counter = int(step_stats_counter) + 1
    step_stats_counter = ('%.' + str(n_digits) + 'd') % _counter
    return step_stats_counter


def change_stats_path(new_value):
    global stats_path
    old_value = stats_path
    stats_path = new_value
    return old_value


def change_stats_filename(new_value):
    global stats_filename
    old_value = stats_filename
    stats_filename = new_value
    return old_value


def clear_two_nn_stats():
    global nn_hits
    global nn_miss
    nn_hits = 0
    nn_miss = 0


def initialize_all(model_parameters: ModelParams, environment: Environment):
    global model_param
    global env

    model_param = model_parameters
    env = environment

    if step_stats:
        initialize_stats()

    # Checks ---
    # Check action codes
    codes = []
    for action in ActionData:
        if action.code not in codes:
            codes.append(action.code)
        else:
            error('(ActionData) Error! Code number of `%s` action is already assigned.' % action.name)
    # ---


def initialize_stats():
    global stats_path
    global stats_filename
    global timestamp

    stats_path, stats_filename, timestamp = stats.create_stats_path_and_filename()
    if verbose_slbo:
        print('Stats file location: ' + stats_path + stats_filename)


class Reservations:
    def __init__(self, people, temp_out):
        if people > max_people:
            error('(get_reservations_profile) csv file exceeded `max_people` (%d) threshold.' % max_people)
        self.__people = people
        self.__temp_out = temp_out

    @property
    def people(self):
        return self.__people

    @property
    def temp_out(self):
        return self.__temp_out


def generate_reservations_file(season=None, save_file=False):
    global counter
    first_hour = reservations_generation_first_hour
    last_hour = reservations_generation_last_hour
    min_temp = reservations_generation_min_temp
    max_temp = reservations_generation_max_temp
    initial_max_temp = random_initial_temp_out_max

    seasons = [None, 'spring', 'summer', 'autumn', 'winter']
    if season not in seasons:
        error('(generate_reservations_file) choose a valid season. You can also choose `None`.')

    steps_in_a_hour = 60 / time_delta
    if not steps_in_a_hour.is_integer():
        error('(generate_reservations_file) choose a valid integer for `time_delta`. '
              'More specifically, a divisor of 60.')
    if not isinstance(first_hour, int) or not isinstance(last_hour, int) or not isinstance(max_people, int):
        error('(generate_reservations_file) `first_hour`, `last_hour` and `max_people` '
              'must be `int` type.')
    if first_hour < 0 or first_hour > 23:
        error('(generate_reservations_file) `first_hour` must be equal or greater than zero '
              'and equal or less than 23.')
    if last_hour < 1 or last_hour > 24:
        error('(generate_reservations_file) `last_hour` must be equal or greater than 1 '
              'and equal or less than 24.')
    if first_hour >= last_hour:
        error('(generate_reservations_file) `first_hour` must be less than `last_hour`.')
    if max_people < 0:
        error('(generate_reservations_file) `max_people` must be equal or greater than zero.')
    if min_temp > max_temp:
        error('(generate_reservations_file) `min_temp` must be equal or less than `max_temp`.')
    if not min_temp <= initial_max_temp <= max_temp:
        error('(generate_reservations_file) `initial_max_temp` must be between `min_temp` and `max_temp`.')

    header = 'step,time,#people,temp_out\n'
    steps = (last_hour - first_hour) * int(steps_in_a_hour)

    string = header
    hour = first_hour
    minute = 0
    consecutive = 0
    people = int(random.uniform(0, max_people))

    if season == 'spring':
        temp = random.uniform(min_temp + 10, initial_max_temp - 5)
    elif season == 'summer':
        temp = random.uniform(initial_max_temp - 10, initial_max_temp)
    elif season == 'autumn':
        temp = random.uniform(min_temp + 10, initial_max_temp - 10)
    elif season == 'winter':
        temp = random.uniform(min_temp, min_temp + 15)
    else:
        temp = random.uniform(min_temp, initial_max_temp)

    for step in range(steps + 1):
        if step != 0:
            if minute + time_delta == 60:
                minute = 0
                hour += 1
            else:
                minute += time_delta

        if consecutive == 0:
            consecutive = random.choice([n for n in range(2, 11)])

        if consecutive != 0:
            consecutive -= 1
            if consecutive == 0:
                people = int(random.uniform(0, max_people))
        else:
            people = int(random.uniform(0, max_people))

        temp = random.gauss(temp, 0.5)
        if temp < min_temp:
            temp = min_temp
        elif temp > max_temp:
            temp = max_temp

        string += '%d,%02d:%02d,%d,%.1f\n' % (step, hour, minute, people, temp)

    timestamp = datetime.utcnow().strftime('%y%m%d_%H%M%S')
    folder = 'datasets/generated_reservations_profiles/%.2d/' % seed
    filename = 'reservations_' + timestamp + '_' + '%.5d' % counter + '.csv'
    filepath = folder + filename
    if save_file:
        counter += 1

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(filepath, 'a') as f:
            f.write(string)

    return string, filepath


def get_reservation_dict(csv_reader):
    global line_count

    line_count = 0
    reservations = defaultdict(dict)
    for row in csv_reader:
        date = row['time']
        people = int(row['#people'])
        temp_out = float(row['temp_out'])

        date_obj = datetime.strptime(date, '%H:%M')
        hour = date_obj.hour
        minute = date_obj.minute

        if hour not in reservations:
            new_dict = defaultdict(dict)
            reservations[hour] = new_dict

        reservations[hour][minute] = Reservations(people, temp_out)
        line_count += 1

    if line_count == 0:
        error('Reservation data is empty.')

    return reservations


def get_reservations_profile(reservations_filepath: str = None, string: str = None):
    if string is not None:
        csv_lines = string.splitlines()
        csv_reader = csv.DictReader(csv_lines)

        return get_reservation_dict(csv_reader)

    try:
        with open(reservations_filepath) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            if verbose_slbo:
                print('Initialization of the room\'s reservations from file.')
                print('File: ' + reservations_filepath)
                print('Column names are ' + str(csv_reader.fieldnames) + '.')

            reservations = get_reservation_dict(csv_reader)

    except FileNotFoundError:
        error('Reservations file does not exist!')

    return reservations


def generate_index_key_references(dictionary):
    referenced_dict = defaultdict(dict)
    ki = defaultdict(int)
    ik = defaultdict(int)
    for i, k in enumerate(dictionary):
        ki[k] = i  # dictionary index_of_key
        ik[i] = k  # dictionary key_of_index

    referenced_dict['ki'] = ki
    referenced_dict['ik'] = ik
    return referenced_dict


def get_referenced_reservations():
    referenced_reservations = defaultdict(dict)

    referenced_reservations['hours'] = generate_index_key_references(reservations)

    referenced_reservations['minutes'] = defaultdict(dict)
    for hour in reservations.keys():
        referenced_dict = generate_index_key_references(reservations[hour])
        referenced_reservations['minutes'][hour] = referenced_dict

    return referenced_reservations


def update_reservations(reservations_filepath: str = None, string: str = None, random_initial_temp_in: bool = True):
    global reservations
    global referenced_reservations
    global initial_temp_in

    if reservations_filepath is None and string is None:
        error('Both `reservations_filepath` and `string` are `None`.')
    if reservations_filepath is not None and string is not None:
        error('Both `reservations_filepath` and `string` are not `None`.')

    reservations = get_reservations_profile(reservations_filepath=reservations_filepath, string=string)
    referenced_reservations = get_referenced_reservations()

    if random_initial_temp_in:
        init_min = random_initial_temp_in_min
        init_max = random_initial_temp_in_max
        out_min = reservations_generation_min_temp
        out_max = reservations_generation_max_temp
        threshold = random_initial_temp_in_threshold

        first_hour = min(reservations.keys())
        first_minute = min(reservations[first_hour].keys())

        initial_temp_in = random.uniform(init_min, init_max)

        if out_min <= reservations[first_hour][first_minute].temp_out < init_min \
                and (init_max - threshold) < initial_temp_in <= init_max:

            initial_temp_in = random.uniform(init_min, init_max - threshold)

        elif init_max < reservations[first_hour][first_minute].temp_out <= out_max \
                and init_min <= initial_temp_in < (init_min + threshold):

            initial_temp_in = random.uniform(init_min + threshold, init_max)

    return reservations, referenced_reservations, initial_temp_in


def initial_state(ignore_utils_values, reservations_: defaultdict = None, initial_temp_in_: float = None):
    global last_hour
    global last_minute

    if reservations_ is None:
        reservations_ = reservations
    if initial_temp_in_ is None:
        initial_temp_in_ = initial_temp_in

    first_hour = min(reservations_.keys())
    first_minute = min(reservations_[first_hour].keys())
    people = reservations_[first_hour][first_minute].people
    if ignore_utils_values:
        co2_low, co2_high = outdoor_co2, ideal_max_co2
        co2 = random.randint(co2_low, co2_high)
        voc_low, voc_high = outdoor_voc, ideal_max_voc
        voc = random.randint(voc_low, voc_high)
    else:
        co2 = initial_co2
        voc = initial_voc
    # First observations
    temp_in = initial_temp_in_
    temp_out = reservations_[first_hour][first_minute].temp_out

    last_hour = max(reservations_.keys())
    last_minute = max(reservations_[last_hour].keys())

    return StateData(first_hour, first_minute, people, co2, voc, temp_in, temp_out), last_hour, last_minute


def set_verbose(new_value: bool):
    global verbose
    verbose = new_value


def error(text: str):
    print('\033[91m' + text + '\033[0m')
    sys.exit(1)


def warning(text: str):
    print('\033[93m' + text + '\033[0m')


if reproducibility:
    with open('reproducibility.pickle', 'rb') as f:
        reproducibility_reservations, reproducibility_temp_in = pickle.load(f)

    if not all_rooms and n_stages > days_per_room:
        error('(utils) `n_stages` parameter is greater than %d.' % days_per_room)

    _room = int(room) - 1

    if not all_rooms:
        reproducibility_reservations = reproducibility_reservations[_room * days_per_room:
                                                                    _room * days_per_room + days_per_room]

    if not all_rooms:
        reproducibility_temp_in = reproducibility_temp_in[_room * days_per_room:
                                                          _room * days_per_room + days_per_room]
