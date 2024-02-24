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
import statistics

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

import utils


def find_file(_folder: str, _type: str, additional: bool, all_files: bool):
    """
    Returns the path of the file containing statistics in the folder
    """
    if 'pickle' in _folder:
        return None
    for root, dirs, files in os.walk(_folder):
        break
    if all_files:
        _files = sorted(files)
        if not _folder.endswith('/'):
            _folder = _folder + '/'
        _files = [_folder + file for file in _files]
        return _files
    for file in files:
        if not additional and file.endswith(_type) and 'additional' not in file:
            if not _folder.endswith('/'):
                _folder = _folder + '/'
            file = _folder + file
            return file
        elif additional and file.endswith(_type) and 'additional' in file:
            if not _folder.endswith('/'):
                _folder = _folder + '/'
            file = _folder + file
            return file
        

def get_filepath_list(folder, _type: str = '.pickle', additional=False, all_files=False):
    """
    Returns a list with all the paths of statistics file in each subfolder
    """
    assert os.path.isdir(folder), '%s is not a directory' %folder
    for root, dirs, files in os.walk(folder):
        break
    folder_list = [folder + _dir for _dir in dirs]
    if all_files:
        filepath_list = [find_file(folder, _type, additional, all_files) for folder in folder_list
                         if find_file(folder, _type, additional, all_files) is not None]
    else:
        filepath_list = [find_file(folder, _type, additional, all_files) for folder in folder_list
                         if find_file(folder, _type, additional, all_files) is not None]

    filepath_list = sorted(filepath_list)

    return filepath_list


def plot_results(MCP_S_filepath_list: list, MCP_M_filepath_list: list, title=None, ppo_stats_path: str = None, slbo_filepath_list: str = None,
                oracle_filepath_list=None, expert_filepath_list=None):
    """
    Creates Figure 2

    Args:
        MCP_S_filepath_list: list of filepath containing statistics of MCP_S test
        MCP_M_filepath_list: list of filepath containing statistics of MCP_M test
        title: title of the figure
        ppo_stats_path: filepath of PPO test statistics
        slbo_filepath_list: list of filepath containing statistics of PPO test
        oracle_filepath_list: list of filepath containing statistics of MCP_O test
        expert_filepath_list: list of filepath containing statistics of MCP_E test
    """
    oracle = True if oracle_filepath_list is not None else False
    expert = True if expert_filepath_list is not None else False

    MCP_S_batches_list = []
    for filepath in MCP_S_filepath_list:
        with open(filepath, 'rb') as f:
            MCP_S_batches_list.append(pickle.load(f))
    
    MCP_M_batches_list = []
    for filepath in MCP_M_filepath_list:
        with open(filepath, 'rb') as f2:
            MCP_M_batches_list.append(pickle.load(f2))

    if oracle:
        oracle_batches_list = []
        for filepath in oracle_filepath_list:
            with open(filepath, 'rb') as f:
                oracle_batches_list.append(pickle.load(f))

    if expert:
        expert_batches_list = []
        for filepath in expert_filepath_list:
            with open(filepath, 'rb') as f:
                expert_batches_list.append(pickle.load(f))

    MCP_S_batches_list_len = len(MCP_S_batches_list)
    assert MCP_S_batches_list_len > 0
    MCP_S_batch_n = list(range(len(MCP_S_batches_list[0])))
    batch_n_offset_1 = list(range(1, len(MCP_S_batches_list[0]) + 1))

    MCP_M_batches_list_len = len(MCP_M_batches_list)
    assert MCP_M_batches_list_len > 0
    MCP_M_batch_n = list(range(len(MCP_M_batches_list[0])))
    batch_n_offset_1_2 = list(range(1, len(MCP_M_batches_list[0]) + 1))


    if oracle:
        oracle_batches_list_len = len(oracle_batches_list)
        assert oracle_batches_list_len > 0

    if expert:
        expert_batches_list_len = len(expert_batches_list)
        assert expert_batches_list_len > 0

    MCP_S_cr_means = [] 
    MCP_M_cr_means = []
    oracle_cr_means = []
    expert_cr_means = []

    MCP_S_co2_error_sums = []
    MCP_S_voc_error_sums = []
    MCP_S_temp_in_error_sums = []

    MCP_M_co2_error_sums = []
    MCP_M_voc_error_sums = [] 
    MCP_M_temp_in_error_sums = []

    expert_co2_error_sums = []
    expert_voc_error_sums = []
    expert_temp_in_error_sums = []

    slbo_cr_means_array = []
    slbo_cr_means = []

    for day in MCP_S_batch_n:
        MCP_S_cr_means_day = [] 
        for room in range(MCP_S_batches_list_len):
            MCP_S_cr_means_day.append(statistics.fmean(MCP_S_batches_list[room][day][0]))
        MCP_S_cr_mean_day = statistics.fmean(MCP_S_cr_means_day)
        MCP_S_cr_means.append(MCP_S_cr_mean_day)

        MCP_M_cr_means_day = [] 
        for room in range(MCP_M_batches_list_len):
            MCP_M_cr_means_day.append(statistics.fmean(MCP_M_batches_list[room][day][0]))
        MCP_M_cr_mean_day = statistics.fmean(MCP_M_cr_means_day)
        MCP_M_cr_means.append(MCP_M_cr_mean_day)

        if oracle:
            oracle_cr_means_day = []
            for room in range(oracle_batches_list_len):
                oracle_cr_means_day.append(statistics.fmean(oracle_batches_list[room][day][1]))
            oracle_cr_mean_day = statistics.fmean(oracle_cr_means_day)
            oracle_cr_means.append(oracle_cr_mean_day)

        if expert:
            expert_cr_means_day = []
            for room in range(expert_batches_list_len):
                expert_cr_means_day.append(statistics.fmean(expert_batches_list[room][day][0]))
            expert_cr_mean_day = statistics.fmean(expert_cr_means_day)
            expert_cr_means.append(expert_cr_mean_day)

        slbo_cr_means_day = []
        for room in range(MCP_S_batches_list_len):
            df = pd.read_csv(slbo_filepath_list[room][day])
            slbo_cr_room_day = sum(df['reward'])
            slbo_cr_means_day.append(slbo_cr_room_day)
        slbo_cr_means_array.append(slbo_cr_means_day)
        slbo_cr_mean_day = statistics.fmean(slbo_cr_means_day)
        slbo_cr_means.append(slbo_cr_mean_day)

        MCP_S_co2_error_day = [] 
        MCP_S_voc_error_day = []
        MCP_S_temp_in_error_day = []

        MCP_M_co2_error_day = []
        MCP_M_voc_error_day = [] 
        MCP_M_temp_in_error_day = [] 

        expert_co2_error_day = []
        expert_voc_error_day = []
        expert_temp_in_error_day = []

        for room in range(MCP_S_batches_list_len):
            MCP_S_co2_error_day.append(sum(MCP_S_batches_list[room][day][2]))
            MCP_S_voc_error_day.append(sum(MCP_S_batches_list[room][day][3]))
            MCP_S_temp_in_error_day.append(sum(MCP_S_batches_list[room][day][4]))

            MCP_M_co2_error_day.append(sum(MCP_M_batches_list[room][day][2]))
            MCP_M_voc_error_day.append(sum(MCP_M_batches_list[room][day][3]))
            MCP_M_temp_in_error_day.append(sum(MCP_M_batches_list[room][day][4]))
            
            if expert:
                expert_co2_error_day.append(sum(expert_batches_list[room][day][2]))
                expert_voc_error_day.append(sum(expert_batches_list[room][day][3]))
                expert_temp_in_error_day.append(sum(expert_batches_list[room][day][4]))

        MCP_S_co2_error_mean_sum = statistics.fmean(MCP_S_co2_error_day) 
        MCP_S_voc_error_mean_sum = statistics.fmean(MCP_S_voc_error_day) 
        MCP_S_temp_in_error_mean_sum = statistics.fmean(MCP_S_temp_in_error_day) 

        MCP_S_co2_error_sums.append(MCP_S_co2_error_mean_sum)
        MCP_S_voc_error_sums.append(MCP_S_voc_error_mean_sum)
        MCP_S_temp_in_error_sums.append(MCP_S_temp_in_error_mean_sum)

        MCP_M_co2_error_mean_sum = statistics.fmean(MCP_M_co2_error_day) 
        MCP_M_voc_error_mean_sum = statistics.fmean(MCP_M_voc_error_day) 
        MCP_M_temp_in_error_mean_sum = statistics.fmean(MCP_M_temp_in_error_day)

        MCP_M_co2_error_sums.append(MCP_M_co2_error_mean_sum)
        MCP_M_voc_error_sums.append(MCP_M_voc_error_mean_sum)
        MCP_M_temp_in_error_sums.append(MCP_M_temp_in_error_mean_sum)


        expert_co2_error_mean_sum = statistics.fmean(expert_co2_error_day)
        expert_voc_error_mean_sum = statistics.fmean(expert_voc_error_day)
        expert_temp_in_error_mean_sum = statistics.fmean(expert_temp_in_error_day)

        expert_co2_error_sums.append(expert_co2_error_mean_sum)
        expert_voc_error_sums.append(expert_voc_error_mean_sum)
        expert_temp_in_error_sums.append(expert_temp_in_error_mean_sum)

    if ppo_stats_path is not None:
        ppo_df = pd.read_csv(ppo_stats_path)
        # header: `profile`, ` day`, ` cumulative_reward`
        ppo_cr_rooms = []
        for room in range(1, MCP_S_batches_list_len + 1):
            ppo_df_room = ppo_df.query('`profile` == @room')
            ppo_cr_room = list(ppo_df_room[[' cumulative_reward']][' cumulative_reward'])
            ppo_cr_rooms.append(ppo_cr_room)
        ppo_cr_array = np.array(ppo_cr_rooms)
        ppo_cr = list(ppo_cr_array.mean(axis=0))

    # Get all deltas
    MCP_S_cr_means_array = [] 
    MCP_M_cr_means_array = []
    for room in range(MCP_S_batches_list_len):
        MCP_S_cr_means_room = []
        MCP_M_cr_means_room = [] 
        for day in MCP_S_batch_n:
            MCP_S_cr_means_room.append(statistics.fmean(MCP_S_batches_list[room][day][0]))
            MCP_M_cr_means_room.append(statistics.fmean(MCP_M_batches_list[room][day][0]))
        MCP_S_cr_means_array.append(MCP_S_cr_means_room)
        MCP_M_cr_means_array.append(MCP_M_cr_means_room)
    MCP_S_cr_means_array = np.array(MCP_S_cr_means_array)
    MCP_M_cr_means_array = np.array(MCP_M_cr_means_array)

    slbo_cr_means_array = np.array(slbo_cr_means_array).T



    if ppo_stats_path is not None:
        deltas = []
        delta_first_days = []
        delta_last_days = []

        for room in range(MCP_M_batches_list_len): 
            for day in MCP_M_batch_n: 
                delta = MCP_M_cr_means_array[room][day] - ppo_cr_array[room][day] 
                deltas.append(delta)

    if expert:
        expert_deltas = []
        expert_delta_first_days = []
        expert_delta_last_days = []

        expert_cr_means_array = []
        for room in range(expert_batches_list_len):
            expert_cr_means_room = []
            for day in MCP_M_batch_n: 
                expert_cr_means_room.append(statistics.fmean(expert_batches_list[room][day][0]))
            expert_cr_means_array.append(expert_cr_means_room)
        expert_cr_means_array = np.array(expert_cr_means_array)

        for room in range(MCP_M_batches_list_len):
            for day in MCP_M_batch_n: 
                delta = MCP_M_cr_means_array[room][day] - expert_cr_means_array[room][day]
                expert_deltas.append(delta)

    if ppo_stats_path is not None and expert:
        ppo_expert_deltas = []
        ppo_expert_delta_first_days = []
        ppo_expert_delta_last_days = []

        for room in range(MCP_M_batches_list_len):
            for day in MCP_M_batch_n:
                delta = ppo_cr_array[room][day] - expert_cr_means_array[room][day]
                ppo_expert_deltas.append(delta)

    if slbo_filepath_list is not None:
        slbo_deltas = []
        for room in range(expert_batches_list_len):
                for day in MCP_M_batch_n:
                    slbo_deltas.append(MCP_M_cr_means_array[room][day] - slbo_cr_means_array[room][day]) 



    plt.rcParams.update({'font.size': 30})
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.style.use('science')

    from scipy import stats
    from math import log10, floor
    
    if ppo_stats_path is not None and expert:
        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.set_size_inches(24.37, 7)
        fig.suptitle(r'\textbf{Distributions of }$\mathbf{\Delta\rho_d}$', x=0.52, y=0.88)

        expert_bins = range(-40, 60, 10)
        ppo_expert_bins = range(-60, 50, 10)
        slbo_expert_bins = range(-60, 50, 10)
        bins = range(-30, 80, 10)

        expert_ttest = stats.ttest_1samp(expert_deltas, popmean=0.0)
        ttest = stats.ttest_1samp(deltas, popmean=0.0)
        slbo_ttest = stats.ttest_1samp(slbo_deltas, popmean=0.0)

        #### MCP_M - MCP_E
        axs[0].hist(expert_deltas, bins=expert_bins, color='skyblue', zorder=3)
        axs[0].axvline(x=0, color='red', linestyle='--', zorder=4, linewidth=4.0)
        all_mean = statistics.fmean(expert_deltas)
        axs[0].axvline(x=all_mean, color='green', linestyle='-', zorder=4, label='mean', linewidth=4.0)
        mean_text = r'$\mathbf{\overline{\Delta \rho_d}:}$' + '\n' + r'\textbf{%.3f}' % all_mean
        axs[0].annotate(mean_text, xy=(all_mean, 250), xytext=(all_mean + 15, 215),
                        arrowprops=dict(arrowstyle='fancy'))
        value, exponent = ('%.3E' % expert_ttest[1]).split('E')
        scientific_notation = r'\textbf{p-value:}' + '\n' + r'$\mathbf{%s * 10^{%s}}$' % (value, exponent)
        axs[0].annotate(scientific_notation, (-43, 375))
        axs[0].set_xticks(expert_bins)
        axs[0].set_xlabel(r'$\mathbf{\Delta \rho_d^{MCP\_M - MCP\_E}}$') 
        axs[0].set_ylabel(r'\textbf{Density}')
        axs[0].set_ylim(0, 460)
        axs[0].grid(zorder=0)

        #### MCP_M - PPO
        axs[1].hist(deltas, bins=bins, color='skyblue', zorder=3)
        axs[1].axvline(x=0, color='red', linestyle='--', zorder=4, linewidth=4.0)
        all_mean = statistics.fmean(deltas)
        axs[1].axvline(x=all_mean, color='green', linestyle='-', zorder=4, label='mean', linewidth=4.0)
        mean_text = r'$\mathbf{\overline{\Delta \rho_d}:}$' + '\n' + r'\textbf{%.3f}' % all_mean
        axs[1].annotate(mean_text, xy=(all_mean, 250), xytext=(all_mean + 15, 215),
                        arrowprops=dict(arrowstyle='fancy'))
        value, exponent = ('%.3E' % ttest[1]).split('E')
        scientific_notation = r'\textbf{p-value:}' + '\n' + r'$\mathbf{%s * 10^{%s}}$' % (value, exponent)
        axs[1].annotate(scientific_notation, (31, 375))
        axs[1].set_xticks(bins)
        axs[1].set_xlabel(r'$\mathbf{\Delta \rho_d^{MCP\_M - PPO}}$') 
        axs[1].set_ylim(0, 460)
        axs[1].grid(zorder=0)


        #### MCP_M - SLBO
        axs[2].hist(slbo_deltas, bins=slbo_expert_bins, color='skyblue', zorder=3)
        axs[2].axvline(x=0, color='red', linestyle='--', zorder=4, linewidth=4.0)
        all_mean = statistics.fmean(slbo_deltas)
        axs[2].axvline(x=all_mean, color='green', linestyle='-', zorder=4, label='mean', linewidth=4.0)
        mean_text = r'$\mathbf{\overline{\Delta \rho_d}:}$' + '\n' + r'\textbf{%.3f}' % all_mean
        axs[2].annotate(mean_text, xy=(all_mean, 250), xytext=(all_mean - 35, 215),
                        arrowprops=dict(arrowstyle='fancy'))
        value, exponent = ('%.3E' % slbo_ttest[1]).split('E')
        scientific_notation = r'\textbf{p-value:}' + '\n' + r'$\mathbf{%s * 10^{%s}}$' % (value, exponent)
        axs[2].annotate(scientific_notation, (-63, 375))
        axs[2].set_xticks(slbo_expert_bins)
        axs[2].set_xlabel(r'$\mathbf{\Delta \rho_d^{MCP\_M - SLBO}}$')
        axs[2].set_ylim(0, 460)
        axs[2].grid(zorder=0)

        fig.tight_layout()

        plt.subplots_adjust(wspace=0.11, hspace=0)
        plt.gcf().text(0.04, 0.1, r'\textbf{a)}')
        plt.gcf().text(0.35, 0.1, r'\textbf{b)}')
        plt.gcf().text(0.665, 0.1, r'\textbf{c)}')

        #fig.show()

        folder = utils.image_folder + utils.nn_folder
        img_filename = 'Figure_2abc' 
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(folder + img_filename + '.png')



    plt.rcParams.update({'font.size': 35})
    plt.rc('xtick', labelsize=35)
    plt.rc('ytick', labelsize=35)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(40, 7)
    if title is not None:
        fig.suptitle(title, y=0.99)

    ax.xaxis.get_major_locator().set_params(integer=True)

    days_ticks = [1] + list(range(10, 101, 10))
    ax.set_xticks(days_ticks)

    MCP_S_batch_n = batch_n_offset_1

    ax.set_title(r'\textbf{Cumulative reward}', fontsize=30)
    ax.axvline(x=1, ymin=0, ymax=0.85, color='black', linestyle='-')
    ax.axvline(x=26, ymin=0, ymax=0.85, color='black', linestyle='-')
    ax.axvline(x=51, ymin=0, ymax=0.85, color='black', linestyle='-')
    ax.axvline(x=76, ymin=0, ymax=0.85, color='black', linestyle='-')
    if expert:
        ax.plot(MCP_S_batch_n, expert_cr_means, label='MCP_E', linestyle=':',  color='blue', linewidth=3.0)
    if ppo_stats_path is not None:
        ax.plot(MCP_S_batch_n, ppo_cr, label='PPO', linestyle='--', color='magenta', linewidth=3.0)
    if slbo_filepath_list is not None:
        ax.plot(MCP_S_batch_n, slbo_cr_means, label='SLBO', linestyle='--', marker='o', color='orange', linewidth=3.0)
    if oracle:
        ax.plot(MCP_S_batch_n, oracle_cr_means, label='MCP_O', linestyle='-.', color='red', linewidth=3.0)
    ax.plot(MCP_S_batch_n, MCP_M_cr_means, label='MCP_M', linestyle='-', color='green', linewidth=3.0) #### 
    ax.plot(MCP_S_batch_n, MCP_S_cr_means, label='MCP_S', linestyle=':', marker='o', color='blueviolet', linewidth=3.0) 

    ax.set_xlim(0, 101)
    ax.set_ylim(45, 120)
    ax.set_xlabel(r'\textbf{Day}')
    ax.set_ylabel(r'$\mathbf{\overline{\rho_d}}$')

    ax.annotate(r'\textbf{Spring}', (11, 50))
    ax.annotate(r'\textbf{Summer}', (35, 50))
    ax.annotate(r'\textbf{Autumn}', (60, 50))
    ax.annotate(r'\textbf{Winter}', (85, 50))

    plt.gcf().text(0.025, 0.15, r'\textbf{d)}')


    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, loc='upper left', ncol=7, bbox_to_anchor=(0, 1.04))
    ax.grid()

    fig.tight_layout()
    #fig.show()

    folder = utils.image_folder + utils.nn_folder
    img_filename = 'Figure_2d'
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(folder + img_filename + '.png')

    data_boxplot = []
    labels_boxplot = []
    if expert:
        data_boxplot.append(expert_cr_means)
        labels_boxplot.append('MCP_E')
    if ppo_stats_path is not None:
        data_boxplot.append(ppo_cr)
        labels_boxplot.append('PPO')
    if slbo_filepath_list is not None:
        data_boxplot.append(slbo_cr_means)
        labels_boxplot.append('SLBO')
    if oracle:
        data_boxplot.append(oracle_cr_means)
        labels_boxplot.append('MCP_O')
    data_boxplot.append(MCP_M_cr_means)
    labels_boxplot.append('MCP_M')
    data_boxplot.append(MCP_S_cr_means)
    labels_boxplot.append('MCP_S')
    

    import matplotlib.patheffects as path_effects


    def add_median_labels(ax, fmt='.1f'):
        lines = ax.get_lines()
        boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
        lines_per_box = int(len(lines) / len(boxes))
        for median in lines[4:len(lines):lines_per_box]:
            x, y = (data.mean() for data in median.get_data())
            # choose value depending on horizontal or vertical plot orientation
            value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
            text = ax.text(x, y - 0.6, f'{value:{fmt}}', ha='center', va='top', #y-0.8
                        fontweight='bold', color='white')
            # create median-colored border around white text for contrast
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ])
    
    #### season box plot 
    season_data = []
    labels_boxplot_season = []
    season_list = ['spring', 'summer', 'autumn', 'winter']
    start = 0
    s_pos = 0
    for i in range(len(expert_cr_means)):
        if (i+1)%25==0:
            if expert:
                season_data.append(expert_cr_means[start:i+1])
                labels_boxplot_season.append('MCP_E')
            if ppo_stats_path is not None:
                season_data.append(ppo_cr[start:i+1])
                labels_boxplot_season.append('PPO')
            if slbo_filepath_list is not None:
                season_data.append(slbo_cr_means[start:i+1])
                labels_boxplot_season.append('SLBO')
            if oracle:
                season_data.append(oracle_cr_means[start:i+1])
                labels_boxplot_season.append('MCP_O')
            season_data.append(MCP_M_cr_means[start:i+1])
            labels_boxplot_season.append('MCP_M')
            season_data.append(MCP_S_cr_means[start:i+1])
            labels_boxplot_season.append('MCP_S')

            start=i+1

            fig_season, ax_season = plt.subplots(1)
            fig_season.set_size_inches(14, 7)
            import seaborn as sns
            ax_season = sns.boxplot(data=season_data, palette=['blue', 'magenta', 'orange', 'red', 'green', 'blueviolet'])
            ax_season.set_xticklabels(labels_boxplot_season)
            add_median_labels(ax_season)
            plt.rcParams.update({'font.size': 35})
            plt.rc('xtick', labelsize=35)
            plt.rc('ytick', labelsize=35)
            plt.style.use('science')
            #plt.show()
            if s_pos == 0:
                plt.gcf().text(0.08, 0.05, r'\textbf{e)}')
            fig_season.savefig(folder + 'Figure_2e_%s' %season_list[s_pos] + '.png')

            s_pos+=1
            season_data = []
            labels_boxplot_season = []
            
    



if __name__ == '__main__':
    base_path = os.path.curdir
    
    oracle_filepath_list = get_filepath_list(os.path.join(base_path,'batch_stats/expert_and_oracle/'))
    expert_filepath_list = get_filepath_list(os.path.join(base_path,'batch_stats/expert_and_oracle/'))
    
    MCP_S_filepath_list = get_filepath_list(os.path.join(base_path,'batch_stats/MCP_S/'))
    MCP_M_filepath_list = get_filepath_list(os.path.join(base_path,'batch_stats/MCP_M/'))

    plot_results(MCP_S_filepath_list=MCP_S_filepath_list,
                 MCP_M_filepath_list=MCP_M_filepath_list,
                 title=None,
                 ppo_stats_path=os.path.join(base_path,'batch_stats/PPO/ppo_train_10_rooms_stats_short.csv'),
                 slbo_filepath_list=get_filepath_list(os.path.join(base_path,'batch_stats/SLBO/'), _type='.csv', all_files=True),
                 oracle_filepath_list=oracle_filepath_list,
                 expert_filepath_list=expert_filepath_list)