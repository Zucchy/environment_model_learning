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

from env_files.model_parameters import ModelParams
import env_files.utils as utils
from env_files.safeplace_env import SafePlaceEnv
from env_files.dataset_generation import *
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["WANDB_START_METHOD"] = "thread"
import numpy as np
from PPO_keras import PPO
import env_files.stats as stats 


def normalize_state(state):
    values = {"people": [0, 50],
    "co2": [400, 2500],
    "voc": [30, 1500],
    "temp_in": [-5, 40],
    "temp_out": [-5, 40]}
   
    state_normalized = []
    for i in range(len(state)):
        # values[list(values.keys())[i]][0] is the min value available for the feature that we are considering
        # in the first case we'll have: values['people'][0] = 0, values['people'][1] = 50 and so on...    
        s_norm = 2*((state[i] - values[list(values.keys())[i]][0]) / (values[list(values.keys())[i]][1] - values[list(values.keys())[i]][0])) - 1
        state_normalized.append(s_norm)

    return np.array([state_normalized])


def pretrain():

    env = SafePlaceEnv()
    approximate_model = DatasetGeneration()
    params = ModelParams(exp_const=None, max_depth=None, rollout_moves=None, iterations=None)
    utils.initialize_all(params, env)

    obs_space = 5
    action_dim = len(list(ActionData))

    seed_nn = 2020     # set random seed if required (0 = no random seed)
    
    ###################### create log file ######################
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/SafePlace_Pretrain/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_cum_Reward = log_dir + '/PPO_SafePlacePretrain_log_' + str(run_num) + ".csv"
    log_stats = log_dir + '/PPO_SafePlacePretrain_stats_log_' + str(run_num) + ".csv"


    ################################## 
    ########## PRE TRAINING ##########
    ################################## 

    # initialize a PPO agent
    ppo_agent = PPO(seed_nn, obs_space, action_dim)

    # logging file
    log_f = open(log_cum_Reward,"w+")
    log_f.write('seed, episode, cumulative_reward\n')

    # logging file
    log_s = open(log_stats,"w+")
    log_s.write('seed, episode, state, action, reward\n')
    seed = list(range(11, 811))
    mean_reward = []

    for episode in range(800):
        stats_list = []
        cum_reward = []
        random.seed(seed[episode])
        string_reservation, _ = utils.generate_reservations_file(save_file=False)
        utils.update_reservations(string=string_reservation, random_initial_temp_in=True)
        state = utils.initial_state()
    
        while not state.is_terminal:
            if utils.step_stats:
                s_stats_time = stats.step_stats_start()

            s = normalize_state(state.s[2:])
            s_action, s_logp, s_vf = ppo_agent.get_action(s)

            action_data = ActionData.get_action(s_action[0])
            next_state = approximate_model.get_next_state(state, action_data, real_env=False)#env.get_next_state(state, action_data)
            reward = env.get_reward(state, action_data, next_state)

            if utils.step_stats:
                stats.step_stats_record_data(state, ActionData.get_action(s_action[0]), reward, s_stats_time, 1, 0)

            stats_list.append({'State': state,
                        'Action':  ActionData.get_action(s_action[0]),
                        'Reward': reward})

            state = next_state

            cum_reward.append(reward[0])    
            ppo_agent.buffer.store(
                        s, 
                        s_action[0], 
                        s_logp[0], 
                        reward[0], 
                        s_vf.squeeze()
                    )

        l_state = s
        l_vf = ppo_agent.vf(l_state)
        ppo_agent.buffer.compute_mc(l_vf, 120)
            
        ppo_agent.update()
        print_avg_reward = round(sum(cum_reward), 2)
        mean_reward.append(print_avg_reward)
        
        print("Episode: {}\t Cumulative reward: {}".format(episode, print_avg_reward))
        log_f.write('{},{},{}\n'.format(seed[episode], episode, print_avg_reward))
        log_f.flush()

        for stat in stats_list:
            log_s.write('{},{},{},{},{}\n'.format(seed[episode], episode, stat['State'], stat['Action'], stat['Reward']))
        log_s.flush()

    ppo_agent.pi.save(f'models_pretrain/pretrain_SafePlaceEnv_keras_actor_seed{seed_nn}_success{print_avg_reward}.h5')
    ppo_agent.vf.save(f'models_pretrain/pretrain_SafePlaceEnv_keras_critic_seed{seed_nn}_success{print_avg_reward}.h5')
    print("Episode: {}\t Mean reward: {}".format(episode, round(np.mean(np.array(mean_reward)),2)))
    log_f.write('\n{},{},{}\n'.format(seed[episode], episode, round(np.mean(np.array(mean_reward)),2)))
    log_f.flush()

    log_f.close()
    log_s.close()


if __name__ == '__main__':
    pretrain()









