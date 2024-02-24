# Online Model Adaptation in Monte Carlo Tree Search Planning



## Description

This is the code used to run a single execution of the algorithms MCP_Real (MCP_R), MCP_Mix (MCP_M) and MCP_Select (MCP_S). 
To run the single algorithms follow the instructions below.


## Installation
This guide is for Linux users only.


### Required interpreters and packages
- CPython 3.9
- PyPy 3.9 
- Execnet
- Tensorflow 2.5.2
- Pandas
- Matplotlib
- SciencePlots
- Scipy

We advise you to create a virtual or conda environment for avoiding potential
compatibility issues, even though it's not mandatory. Skip this sub-paragraph
if you don't intend to use a conda environment.


### Create a conda environment
First, run the following commands on your terminal to install Miniconda:
<pre>
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc                        # Restart the terminal
conda -V                                # Print conda version
conda deactivate                        # To disable <i>base</i> env if active
</pre>

Then, create a conda environment (we named ours `tf-2.5.2`) choosing Python 3.9:
<pre>
conda create --name tf-2.5.2 python=3.9
conda activate tf-2.5.2                  # To enable <i>tf-2.5.2</i> env
</pre>


### Install the required packages

##### Install the packages
Make sure your conda/virtual env is active. If you are not using them, beware that
your packages will be overwritten if necessary.

Install the required packages:
<pre>
pip install --upgrade pip
pip install tensorflow==2.5.2 numpy==1.19.3 pandas==1.4.4 execnet matplotlib SciencePlots==1.0.9 scipy seaborn
sudo apt install texlive-full
sudo apt install cm-super dvipng
</pre>


##### Install PyPy interpreter
Run the following commands to download and extract to your desired location
(i.e., `/your/pypy/path`) PyPy 3.9 archive:
<pre>
curl https://downloads.python.org/pypy/pypy3.9-v7.3.9-linux64.tar.bz2 -o pypy.tar.bz2
tar -xf pypy.tar.bz2 -C /your/pypy/path    # Replace <i>/your/pypy/path</i> with a path you choose
</pre>


##### Edit `code/MCP/utils.py` file
In order to execute our program in experiment mode, change the value of
`execnet_interpreter_path` at line 70 with your PyPy interpreter path.

Assuming you didn't change the folder structure of the extracted archive,
you need to replace line 70 with  
`execnet_interpreter_path = '/your/pypy/path/pypy3.9-v7.3.9-linux64/bin/pypy3.9'`



## Usage

### Variant MCP_R
Clone the folder `MCP` in `code` and name it `MCP_R`. In `MCP_R/utils.py` set the value of following variables:
<pre>
seed = 582039                               # line 33, you can change the seed if you want
tf_seed = 482723                            # line 35, you can change the seed if you want
temperature_seed =1234						# line 41, you can change the seed if you want
room = '01'                                 # line 44 
use_two_neural_networks = 'False'           # line 86
optimize_dataset_between_batches = 'False'  # line 87
</pre>
All other variables are already set to the correct value. 

To run MCP_R, execute the following commands in a new terminal from the active working directory `MCP_R`(make sure you are inside `MCP_R` folder):
<pre>
conda deactivate                  # To disable <i>base</i> env if active
export CUDA_VISIBLE_DEVICES=""    # To avoid using GPU devices instead of CPU
conda activate tf-2.5.2           # Activate the env
python3 execnet_tf.py             # Execute script <i>execnet_tf.py</i>
</pre>

This will create in MCP_R two new folders named `batch_stats` and `step_stats` containing the results.
In `batch_stats` you will find a .pickle file containing 100 rows and 5 columns. Each rows contains the following elements: cumulative reward of MCP_R, cumulative reward of the oracle, absolute error in computing CO2 concentration (120 values), absolute error in computing the VOC concentration (120 values), and absolute error in computing internal temperature (120 values).
In `step_stats` going into subfolders you will find 100 .csv files, one per day. Each file contains 120 rows, one per time step, and 14 columns indicating the content of each element in the row. The columns are: 'time',  'people', 'co2', 'voc', 'temp_in', 'temp_out', 'action', 'reward', 'air_quality_reward', 'comfort_reward', 'energy_reward', 's_stats_time', 'sim_done', 'sim_wasted'.


### Variant MCP_M
Clone the folder `MCP` in `code` and name it `MCP_M`. In `MCP_M/utils.py` set the value of following variables:
<pre>
seed = 582039                               # line 33, you can change the seed if you want
tf_seed = 482723                            # line 35, you can change the seed if you want
temperature_seed =1234						# line 41, you can change the seed if you want
room = '01'                                 # line 44 
use_two_neural_networks = 'False'           # line 86
optimize_dataset_between_batches = 'True'   # line 87
people_threshold = 60                       # line 125, you can change the value if you want
co2_threshold = 800                         # line 126, you can change the value if you want
voc_threshold = 800                         # line 127, you can change the value if you want
temp_in_threshold = 25                      # line 128, you can change the value if you want
temp_out_threshold = 25                     # line 129, you can change the value if you want
</pre>
All other variables are already set to the correct value. 

To run MCP_M, execute the following commands in a new terminal from the active working directory `MCP_M`(make sure you are inside `MCP_M` folder):
<pre>
conda deactivate                  # To disable <i>base</i> env if active
export CUDA_VISIBLE_DEVICES=""    # To avoid using GPU devices instead of CPU
conda activate tf-2.5.2           # Activate the env
python3 execnet_tf.py             # Execute script <i>execnet_tf.py</i>
</pre>

This will create in MCP_M two new folders named `batch_stats` and `step_stats` containing the results.
In `batch_stats` you will find a .pickle file containing 100 rows and 5 columns. Each rows contains the following elements: cumulative reward of MCP_M, cumulative reward of the oracle, absolute error in computing CO2 concentration (120 values), absolute error in computing the VOC concentration (120 values), and absolute error in computing internal temperature (120 values).
In `step_stats` going into subfolders you will find 100 .csv files, one per day. Each file contains 120 rows, one per time step, and 14 columns indicating the content of each element in the row. The columns are: 'time',  'people', 'co2', 'voc', 'temp_in', 'temp_out', 'action', 'reward', 'air_quality_reward', 'comfort_reward', 'energy_reward', 's_stats_time', 'sim_done', 'sim_wasted'.


### Variant MCP_S
Clone the folder `MCP` in `code` and name it `MCP_S`. In `MCP_S/utils.py` set the value of following variables:
<pre>
seed = 582039                               # line 33, you can change the seed if you want
tf_seed = 482723                            # line 35, you can change the seed if you want
temperature_seed =1234						# line 41, you can change the seed if you want
room = '01'                                 # line 44 
use_two_neural_networks = 'True'            # line 86
optimize_dataset_between_batches = 'False'  # line 87
people_threshold=8                          # line 125, you can change the value if you want
co2_threshold=300                           # line 126, you can change the value if you want
voc_threshold=300                           # line 127, you can change the value if you want
temp_in_threshold=3                         # line 128, you can change the value if you want
temp_out_threshold=3                        # line 129, you can change the value if you want
</pre>
All other variables are already set to the correct value. 

To run MCP_S, execute the following commands in a new terminal from the active working direcory `MCP_S`(make sure you are inside `MCP_S` folder):
<pre>
conda deactivate                  # To disable <i>base</i> env if active
export CUDA_VISIBLE_DEVICES=""    # To avoid using GPU devices instead of CPU
conda activate tf-2.5.2           # Activate the env
python3 execnet_tf.py             # Execute script <i>execnet_tf.py</i>
</pre>

This will create in MCP_S two new folders named `batch_stats` and `step_stats` containing the results.
In `batch_stats` you will find a .pickle file containing 100 rows and 5 columns. Each rows contains the following elements: cumulative reward of MCP_S, cumulative reward of the oracle, absolute error in computing CO2 concentration (120 values), absolute error in computing the VOC concentration (120 values), and absolute error in computing internal temperature (120 values).
In `step_stats` going into subfolders you will find 100 .csv files, one per day. Each file contains 120 rows, one per time step, and 14 columns indicating the content of each element in the row. The columns are: 'time',  'people', 'co2', 'voc', 'temp_in', 'temp_out', 'action', 'reward', 'air_quality_reward', 'comfort_reward', 'energy_reward', 's_stats_time', 'sim_done', 'sim_wasted'.


# References
M. Zuccotto, E. Fusa, A. Castellini, and A. Farinelli. "Online model adaptation in Monte Carlo tree search planning". (Submitted)

# Contacts
For any information/bugfix/issue contact Edoardo Fusa at edoardo.fusa@gmail.com
