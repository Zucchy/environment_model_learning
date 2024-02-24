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


class NNPyPy:
    """
    Class used by `SafePlaceEnvNNPyPy` to predict the next state in the MDP. Every instance of `NNPyPy` class is created
    in `neural_network.py`.
    """
    def __init__(self, w_py: list, is_normalized: bool) -> None:
        """
        Method called after the creation of the object. It just saves the data.

        Args:
            w_py: neural network weights as a list.
            is_normalized:
        """
        self.weights = w_py
        self.weights_length = len(w_py)
        self.is_normalized = is_normalized

    def model_inference(self, x: list) -> list:
        """
        This function uses the neural network weights to predict the next state of the MDP. The results are used to
        create a new state node in the MCTS tree.

        Args:
            x: input of the neural network.

        Returns: the output of the neural network as a list.
        """
        y = None
        for i in range(self.weights_length):
            if self.is_normalized and i == 0:
                y = m_div(m_sub(x, self.weights[i][0]), m_sqrt(self.weights[i][1]))
                continue
            y = m_add(m_mul(y, self.weights[i][0]), self.weights[i][1])
            if i == self.weights_length - 1:
                break
            y = relu(y)

        return y


def m_add(A: list, B: list) -> list:
    """
    Addition between two lists.

    Args:
        A: first list.
        B: second list.

    Returns: `A` list modified.
    """
    lenA = len(A)

    for i in range(lenA):
        A[i] += B[i]

    return A


def m_sub(A: list, B: list) -> list:
    """
    Subtraction between two lists.

    Args:
        A: first list.
        B: second list.

    Returns: a new list.
    """
    lenA = len(A)

    C = [0] * lenA
    for i in range(lenA):
        C[i] = A[i] - B[i]

    return C


def m_mul(A: list, B: list) -> list:
    """
    Multiplication between two lists (using matrix multiplication rules).

    Args:
        A: first list.
        B: second list.

    Returns: a new list.
    """
    lenA = len(A)
    colsB = len(B[0])

    C = [0] * colsB
    for i in range(colsB):
        total = 0
        for j in range(lenA):
            total += A[j] * B[j][i]
        C[i] = total

    return C


def m_div(A: list, B: list) -> list:
    """
    Division between two lists.

    Args:
        A: first list.
        B: second list.

    Returns: `A` list modified.
    """
    lenA = len(A)

    for i in range(lenA):
        A[i] /= B[i]

    return A


def m_sqrt(A: list) -> list:
    """
    Square root of a list.

    Args:
        A: the list.

    Returns: a new list.
    """
    length = len(A)

    C = [0] * length
    for i in range(length):
        C[i] = math.sqrt(A[i])

    return C


# Other options for ReLU
# i)
"""
if A[i] < 0:
    A[i] = 0
"""

# ii)
"""
A[i] *= (A[i] > 0)
"""


def relu(A: list) -> list:
    """
    ReLU function applied on a list.

    Args:
        A: the list.

    Returns: `A` list modified.
    """
    length = len(A)

    for i in range(length):
        A[i] = (abs(A[i]) + A[i]) / 2

    return A
