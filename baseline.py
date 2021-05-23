#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 10/05/21 11:41 AM
@description:  
@version: 1.0
"""


import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
import time
import json
import argparse
from models.AE import AE
import visdom


data_path = './data/data_processed'
info_path = './data/info'
save_path = './results'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Baseline:

    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        print('> Training arguments:')
        for arg in vars(args):
            print('>>> {}: {}'.format(arg, getattr(args, arg)))
        # Dataset
        white_noise = dp.ReshapeDataset(white_noise=args.dataset,
                                        data_path=data_path,
                                        data_source=args.data_source,
                                        len_seg=args.len_seg)
        trainset, _ = white_noise()
        trainset_eos = white_noise.decoder_input_init(trainset,
                                                      last_time_step=args.last_time_step,
                                                      is_trainset=True
                                                      )
        dataset = dp.Trainset(trainset, trainset_eos)
        encoder_inputs_train, decoder_inputs_train, decoder_outputs_train = dataset()
        self.data_loader = DataLoader(dp.SignalDataset(encoder_inputs_train,
                                                       decoder_inputs_train,
                                                       decoder_outputs_train
                                                       ),
                                      batch_size=args.batch_size,
                                      shuffle=False
                                      )
        self.spots = np.load('{}/spots.npy'.format(info_path))
        self.model = AE(args).to(device)  # AutoEncoder
        self.criterion = nn.MSELoss()
        # self.vis = visdom.Visdom(port=8097,
        #                          env='{}'.format(self.file_name()),
        #                          log_to_filename='{}/visualization/{}.log'.
        #                          format(save_path, self.file_name())
        #                          )
        # plt.figure(figsize=(15, 15))

    def weights_init(self):
        for op in self.model.modules():
            if isinstance(op, nn.Linear):
                nn.init.xavier_uniform_(op.weight)

    def select_optimizer(self):
        assert self.args.optimizer == 'SGD' or 'Adam'
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.args.learning_rate
                                  )
        else:
            optimizer = optim.Adam(self.model.parameters(),
                                   betas=(0.5, 0.999),
                                   lr=self.args.learning_rate
                                   )
        return optimizer

    def file_name(self):
        return '{}_{}_{}_{}_{}_{}_{}'.format(self.args.model,
                                             self.args.len_seg,
                                             self.args.optimizer,
                                             self.args.learning_rate,
                                             self.args.num_epoch,
                                             self.args.dim_feature,
                                             self.args.batch_size,
                                             )

    def train(self):
        optimizer = self.select_optimizer()  # Select optimizer
        self.weights_init()  # Initialize weights
        best_loss = 100.
        best_epoch = 1
        lh = {}
        losses_all = []
        for epoch in range(self.args.num_epoch):
            losses = 0
            t_0 = time.time()
            for enc_input_batch, _, _ in self.data_loader:
                enc_input_batch = enc_input_batch.squeeze(2).to(device)
                x = enc_input_batch[:, 0: 2 * self.args.len_seg]
                z, x_hat = self.model(x)
                loss = self.criterion(x, x_hat)
                losses += loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)  # Gradient clip
                optimizer.step()
            losses_all.append(losses.item())
            t_1 = time.time()
            if losses.item() < best_loss:
                best_loss = losses.item()
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(),
                           '{}/models/{}.model'.format(save_path, self.file_name())
                           )
            # self.show_loss(losses, epoch)
            # self.show_reconstruction(epoch)
            print('\033[1;31mEpoch: {}\033[0m\t'
                  '\033[1;32mLoss: {:5f}\033[0m\t'
                  '\033[1;33mTime cost: {:2f}s\033[0m'
                  .format(epoch + 1, losses, t_1 - t_0))
        # plt.close()
        lh['Loss'] = losses_all
        lh['Min loss'] = best_loss
        lh['Best epoch'] = best_epoch
        lh = json.dumps(lh, indent=2)
        with open('{}/learning history/{}.json'.format(save_path, self.file_name()), 'w') as f:
            f.write(lh)
        self.show_results()
        plt.savefig('./results/visualization/{}.png'.format(self.file_name()))
        plt.show()

    def show_loss(self, loss, epoch):
        self.vis.line(Y=np.array([loss.item()]), X=np.array([epoch + 1]),
                      win='Train loss',
                      opts=dict(title='Train loss'),
                      update='append'
                      )

    def show_reconstruction(self, epoch, seg_idx=10):
        plt.clf()
        encoder_inputs = self.data_loader.dataset.encoder_inputs[:, 0:2 * self.args.len_seg, :].squeeze(2)
        num_seg = int(encoder_inputs.shape[0] / len(self.spots))
        spots_l1, spots_l2 = np.hsplit(self.spots, 2)
        for i, (spot_l1, spot_l2) in enumerate(zip(spots_l1, spots_l2)):
            # 1 sensors
            plt.subplot(int(len(self.spots) / 2), 2, 2 * i + 1)
            x = encoder_inputs[i * num_seg + seg_idx].unsqueeze(0)
            x = x.to(device)
            plt.plot(x.view(-1).detach().cpu().numpy(), label='original')
            plt.title('AC-{}-{}'.format(spot_l1, seg_idx))
            z, x_hat = self.model(x)
            plt.plot(x_hat.view(-1).detach().cpu().numpy(), label='reconstruct')
            plt.axvline(x=self.args.len_seg - 1, ls='--', c='k')
            plt.legend(loc='upper center')
            # R sensors
            plt.subplot(int(len(self.spots) / 2), 2, 2 * (i + 1))
            x = encoder_inputs[(i + 6) * num_seg + seg_idx].unsqueeze(0)
            x = x.to(device)
            plt.plot(x.view(-1).detach().cpu().numpy(), label='original')
            plt.title('AC-{}-{}'.format(spot_l2, seg_idx))
            z, x_hat = self.model(x)
            plt.plot(x_hat.view(-1).detach().cpu().numpy(), label='reconstruct')
            plt.axvline(x=self.args.len_seg - 1, ls='--', c='k')
            plt.legend(loc='upper center')
        plt.subplots_adjust(hspace=0.5)
        self.vis.matplot(plt, win='Reconstruction', opts=dict(title='Epoch: {}'.format(epoch + 1)))

    def show_results(self, seg_idx=10):
        path = '{}/models/{}.model'.format(save_path,
                                           self.file_name()
                                           )
        model = AE(self.args).to(device)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.eval()
        fig, axs = plt.subplots(nrows=int(len(self.spots) / 2), ncols=2, figsize=(20, 20))
        encoder_inputs = self.data_loader.dataset.encoder_inputs[:, 0:2 * self.args.len_seg, :].squeeze(2)
        num_seg = int(encoder_inputs.shape[0] / len(self.spots))
        spots_l1, spots_l2 = np.hsplit(self.spots, 2)
        for i, (spot_l1, spot_l2) in enumerate(zip(spots_l1, spots_l2)):
            # 1 sensors
            x = encoder_inputs[i * num_seg + seg_idx].unsqueeze(0)
            x = x.to(device)
            axs[i][0].plot(x.view(-1).detach().cpu().numpy(), label='original')
            axs[i][0].set_title('AC-{}-{}'.format(spot_l1, seg_idx))
            _, x_hat = self.model(x)
            axs[i][0].plot(x_hat.view(-1).detach().cpu().numpy(), label='reconstruct')
            axs[i][0].axvline(x=self.args.len_seg - 1, ls='--', c='k')
            axs[i][0].legend(loc='upper center')
            # R sensors
            x = encoder_inputs[(i + 6) * num_seg + seg_idx].unsqueeze(0)
            x = x.to(device)
            axs[i][1].plot(x.view(-1).detach().cpu().numpy(), label='original')
            axs[i][1].set_title('AC-{}-{}'.format(spot_l2, seg_idx))
            _, x_hat = self.model(x)
            axs[i][1].plot(x_hat.view(-1).detach().cpu().numpy(), label='reconstruct')
            axs[i][1].axvline(x=self.args.len_seg - 1, ls='--', c='k')
            axs[i][1].legend(loc='upper center')
        # plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()


if __name__ == '__main__':
    # Hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WN1', type=str)
    parser.add_argument('--data_source', default='denoised', type=str)
    parser.add_argument('--model', default='MLP', type=str)
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--last_time_step', action='store_true', default=False)
    parser.add_argument('--dim_feature', default=128, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--seed', default=23, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    args = parser.parse_args()
    exp = Baseline(args)
    exp.train()
