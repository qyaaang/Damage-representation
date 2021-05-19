#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 19/05/21 5:52 PM
@description:  
@version: 1.0
"""


import torch
import numpy as np
import data_processing as dp
import argparse
import json
from models.AE import AE


data_path = './data/data_processed'
info_path = './data/info'
save_path = './results'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DamageDetection:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        print('{} detection...'.format(args.dataset))
        # Dataset
        white_noise = dp.ReshapeDataset(white_noise=args.dataset,
                                        data_path=data_path,
                                        data_source=args.data_source,
                                        len_seg=args.len_seg)
        _, testset = white_noise()
        testset_eos = white_noise.decoder_input_init(testset,
                                                     last_time_step=args.last_time_step,
                                                     is_trainset=True
                                                     )
        dataset = dp.Testset(testset, testset_eos)
        self.encoder_inputs_test, self.decoder_inputs_test, self.decoder_outputs_test = dataset()
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.model = AE(args).to(device)  # Seq2Seq model

    def __call__(self, *args, **kwargs):
        self.test()

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}_{}'.format(self.args.model,
                                             self.args.len_seg,
                                             self.args.optimizer,
                                             self.args.learning_rate,
                                             self.args.num_epoch,
                                             self.args.dim_feature,
                                             self.args.batch_size,
                                             )

    def test(self):
        path = '{}/models/{}.model'.format(save_path,
                                           self.file_name()
                                           )
        self.model.load_state_dict(torch.load(path, map_location=torch.device(device)))  # Load AutoEncoder
        self.model.eval()
        with torch.no_grad():
            for i, spot in enumerate(self.spots):
                x = self.encoder_inputs_test[i].squeeze(2).to(device)
                x = x[:, 0: 2 * self.args.len_seg]
                _, x_hat = self.model(x)
                loss = ((x_hat - x) ** 2).mean()
                print(loss.item(), spot)


def main():
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WN2', type=str)
    parser.add_argument('--data_source', default='denoised', type=str)
    parser.add_argument('--model', default='MLP', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--last_time_step', action='store_true', default=False)
    parser.add_argument('--dim_feature', default=128, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epoch', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    args = parser.parse_args()
    detector = DamageDetection(args)
    detector()


if __name__ == '__main__':
    main()
