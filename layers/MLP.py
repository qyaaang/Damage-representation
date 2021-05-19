#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 10/05/21 11:19 AM
@description:  
@version: 1.0
"""


from torch import nn


class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(args.len_seg * 2, 512),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(512, 256),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(256, 128),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(128, args.dim_feature),
                                     nn.LeakyReLU(0.2, inplace=True)
                                     )
        self.decoder = nn.Sequential(nn.Linear(args.dim_feature, 128),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(128, 256),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(256, 512),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(512, args.len_seg * 2),
                                     nn.Tanh()
                                     )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat
