#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 7/05/21 5:23 PM
@description:  
@version: 1.0
"""


from torch import nn


class MT_GRU(nn.Module):

    def __init__(self, args):
        super(MT_GRU, self).__init__()
        self.encoder = nn.GRU(input_size=args.input_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              dropout=0.5
                              )
        self.decoder = nn.GRU(input_size=args.input_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              dropout=0.5
                              )
        self.fc = nn.Linear(args.hidden_size, args.embedding_size)

    def forward(self, encoder_inputs, encoder_hidden, decoder_inputs):
        # h_t : [batch_size, num_layers(=1) * num_directions(=1), n_hidden]
        _, h_t = self.encoder(encoder_inputs, encoder_hidden)
        # outputs : [batch_size, dim_seq+1, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.decoder(decoder_inputs, h_t)
        # y: [batch_size, dim_seq+1, dim_embedding(=1)]
        y = self.fc(outputs)
        return y, h_t
