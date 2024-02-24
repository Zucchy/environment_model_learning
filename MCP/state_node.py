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

import math
import random
from collections import defaultdict

from action_node import ActionNode
from state_data import StateData
from action_data import ActionData


class StateNode:
    """
    A state node, i.e. an object representing a state in the MCTS tree linked to the subsequent actions.
    """
    def __init__(self, s: StateData) -> None:
        """
        Method called after the creation of the object. It just saves the data.

        Args:
            s: `StateData` object (i.e., the state).
        """
        self.state = s

        # Number of visits
        self.n = 0
        # The total value of the node
        self.v = 0

        # Child nodes (actions)
        self.actions = defaultdict(ActionNode)
        # Create all the action nodes
        self.expand_action_nodes()

    def expand_action_nodes(self) -> None:
        """
        Expands the tree with all the possible legal action nodes.
        """
        for a in ActionData:
            self.actions[a.code] = ActionNode(a)

    def select_action(self) -> int:
        """
        Select the next action following UCB rules.

        Returns: the action code associated with the `ActionData` enumeration.
        """
        actions_to_simulate = []
        for action_code, an in self.actions.items():
            if an.n == 0:
                actions_to_simulate.append(action_code)

        if len(actions_to_simulate) != 0:
            return random.choice(actions_to_simulate)
        else:
            ucb_values = defaultdict(float)
            for action_code, an in self.actions.items():
                ucb_values[action_code] = an.q() + utils.model_param.exp_const * \
                                           math.sqrt(math.log(self.n) / an.n)

            return max(ucb_values, key=ucb_values.get)

    def simulate_from_state(self, depth=0) -> float:
        """
        Recursive function that executes the MCTS simulation from the current state.

        Args:
            depth: depth of the current state node.

        Returns: the reward of the episode starting from the current state.
        """
        action_code = self.select_action()

        reward = self.actions[action_code].simulate_from_action(self, depth=depth)

        self.n += 1
        self.v += reward

        return reward

    @staticmethod
    def rollout(s: StateData, depth, rollout_depth=0) -> float:
        """
        Recursive static (i.e., does not depend on any `StateNode` object) function that executes MCTS rollout from the
        given state.

        Args:
            s: current `StateData` object. Notice that in the `rollout` function there is no need to use tree nodes.
                This expedient significantly speeds up simulation time.
            depth: current depth relating the tree (it's only a relation because the tree will not be expanded in this
                function). This value is only used to stop simulating when the `utils.model_param.max_depth` is reached
                (we are in a finite horizon MDP).
            rollout_depth: integer value that represents how many times the `rollout` function has been called (this
                number is only related to the current rollout phase; in fact, every time `rollout_depth` gets reset to
                0). In our experiments `utils.model_param.rollout_moves` is set to infinity.

        Returns: the total reward obtained in the rollout phase starting from the current state.
        """

        a = random.choice(list(ActionData))

        if depth >= utils.model_param.max_depth or s.is_terminal or \
                rollout_depth >= utils.model_param.rollout_moves:
            reward = 0
        else:
            next_s, reward = utils.env.simulate(s, a)
            rollout_reward = StateNode.rollout(next_s, depth + 1, rollout_depth + 1)

            reward = reward + rollout_reward

        return reward
