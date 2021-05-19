#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/05/21 5:24 PM
@description:  
@version: 1.0
"""


from torch import nn
from layers.MT_RNN import MT_RNN
from layers.MT_LSTM import MT_LSTM
from layers.MT_GRU import MT_GRU


model_classes = {'RNN': MT_RNN,
                 'LSTM': MT_LSTM,
                 'GRU': MT_GRU
                 }


class Seq2Seq(nn.Module):

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.model = model_classes[args.model](args)

    def forward(self, encoder_inputs, encoder_hidden, encoder_cell, decoder_inputs):
        if self.args.model == 'LSTM':
            return self.model(encoder_inputs, encoder_hidden, encoder_cell, decoder_inputs)
        else:
            return self.model(encoder_inputs, encoder_hidden, decoder_inputs)
