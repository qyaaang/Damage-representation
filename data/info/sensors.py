#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 4/05/21 10:26 AM
@description:  
@version: 1.0
"""


import numpy as np

levels = ['1', 'R']
locations = {'1': ['101', '102', '401', '402',
                   '103', '301', '403'],
             'R': ['101', '102', '401', '402',
                   '103', '313', '403']
             }
dirs = ['N', 'E']
spots = []
sensors = []
for level in levels:
    for location in locations[level]:
        spot = '{}-{}'.format(level, location)
        spots.append(spot)
        for d in dirs:
            sensor_name = 'AC{}{}{}'.format(level, d, location)
            sensors.append(sensor_name)
spots = np.array(spots)
sensors = np.array(sensors)
np.save('./sensors.npy', sensors)
np.save('./spots.npy', spots)
