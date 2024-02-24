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

n_reservations = 100

for i in range(n_reservations):
    if 0 <= i <= 24:
        season = 'spring'
    elif 25 <= i <= 49:
        season = 'summer'
    elif 50 <= i <= 74:
        season = 'autumn'
    elif 75 <= i <= 99:
        season = 'winter'
    else:
        season = ''
    utils.generate_reservations_file(season=season, save_file=True)
