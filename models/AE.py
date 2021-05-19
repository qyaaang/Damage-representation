#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 10/05/21 11:34 AM
@description:  
@version: 1.0
"""


from torch import nn
from layers.MLP import MLP


class AE(nn.Module):

    def __init__(self, args):
        super(AE, self).__init__()
        self.model = MLP(args)

    def forward(self, x):
        z, x_hat = self.model(x)
        return z, x_hat
