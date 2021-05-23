#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 22/05/21 11:00 AM
@description:  
@version: 1.0
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ReshapeDataset2:

    def __init__(self, white_noise, data_path, data_source, len_seg):
        self.white_noise = white_noise
        self.data = np.load('{0}/{1}/{2}/{3}_{1}.npy'.format(data_path, data_source, len_seg, white_noise))
        self.len_seg = len_seg  # Length of sequence
        self.num_sensor = self.data.shape[0]
        self.num_channel = self.data.shape[1]
        self.num_seg = self.data.shape[2]  # Number of sequence
        self.num_feature = self.data.shape[3]  # Length of sequence

    def __call__(self, *args, **kwargs):
        print('Preparing {} dataset...'.format(self.white_noise))
        trainset, testset = self.reshape_trainset_2(), self.reshape_testset_2()
        trainset = torch.tensor(trainset, dtype=torch.float32)
        testset = torch.tensor(testset, dtype=torch.float32)
        for i in range(trainset.size(0)):
            trainset[i] = self.normalization(trainset[i], dim=0)
        for i in range(testset.size(0)):
            for j in range(testset.size(1)):
                testset[i, j] = self.normalization(testset[i, j], dim=0)
        return trainset, testset

    def reshape_trainset_2(self):
        """
        Reshape the trainset
        [14, 2, num_seq, len_seq] => [14 * num_seq, len_seq, 2]
        len_seq = length of each sequence
        :return:
        """
        trainset = self.reshape_testset_2()
        # [14, num_seq, len_seq, 2] => [14 * num_seq, len_seq, 2]
        trainset = trainset.reshape((self.num_sensor * self.num_seg, self.num_feature, -1))
        return trainset

    def reshape_testset_2(self):
        """
        Reshape the trainset
        [14, 2, num_seq, len_seq] => [14, num_seq, len_seq, 2]
        len_seq = length of each sequence
        :return:
        """
        return self.data.transpose((0, 2, 3, 1))

    @staticmethod
    def normalization(original_data, dim=1):
        """
        Normalize the data to [-1, 1]
        :param original_data:
        :param dim:
        :return:
        """
        if dim == 1:
            d_min = torch.min(original_data, dim=dim)[0]
            for idx, val in enumerate(d_min):
                if val < 0:
                    original_data[idx, :] += torch.abs(d_min[idx])
        else:
            d_min = torch.min(original_data, dim=dim)[0]
            for idx, val in enumerate(d_min):
                if val < 0:
                    original_data[:, idx] += torch.abs(d_min[idx])
        d_min = torch.min(original_data, dim=dim)[0]
        d_max = torch.max(original_data, dim=dim)[0]
        dst = d_max - d_min
        if d_min.shape[0] == original_data.shape[0]:
            d_min = d_min.unsqueeze(1)
            dst = dst.unsqueeze(1)
        else:
            d_min = d_min.unsqueeze(0)
            dst = dst.unsqueeze(0)
        norm_data = torch.sub(original_data, d_min).true_divide(dst)
        norm_data = (norm_data - 0.5).true_divide(0.5)
        return norm_data


class Trainset2:

    def __init__(self, signal, num_sensors=14):
        self.signal = signal  # [num_sensors * num_seg, len_seq, 2]
        self.num_sensors = num_sensors
        self.num_seg_all = signal.size(0)
        self.num_seg = int(self.num_seg_all / num_sensors)
        self.len_seq = signal.size(1)

    def __call__(self, *args, **kwargs):
        return self.encoding_trainset()

    def encoding_trainset(self):
        encoder_inputs = np.zeros((self.num_seg_all, self.len_seq + 1, 2))
        decoder_inputs = np.zeros((self.num_seg_all, self.len_seq + 1, 2))
        decoder_outputs = np.zeros((self.num_seg_all, self.len_seq + 1, 2))
        encoder_inputs[:, :-1, :] = self.signal
        encoder_inputs[:, -1, :] = torch.zeros(1, 2)  # positional encoding of E
        decoder_inputs[:, 0, :] = torch.zeros(1, 2)  # positional encoding of S
        decoder_inputs[:, 1:, :] = self.signal
        decoder_outputs[:, :-1, :] = self.signal
        decoder_outputs[:, -1, :] = torch.zeros(1, 2)  # positional encoding of E
        return torch.Tensor(encoder_inputs), torch.Tensor(decoder_inputs), torch.Tensor(decoder_outputs)


class Testset2:

    def __init__(self, signal):
        self.signal = signal  # [num_sensors, num_seg, len_seq, 2]
        self.num_sensors = signal.size(0)
        self.num_seg = signal.size(1)
        self.len_seq = signal.size(2)

    def __call__(self, *args, **kwargs):
        return self.encoding_testset()

    def encoding_testset(self):
        encoder_inputs = np.zeros((self.num_sensors, self.num_seg, self.len_seq + 1, 2))
        decoder_inputs = np.zeros((self.num_sensors, self.num_seg, self.len_seq + 1, 2))
        decoder_outputs = np.zeros((self.num_sensors, self.num_seg, self.len_seq + 1, 2))
        for i in range(self.num_sensors):
            encoder_inputs[i, :, :-1, :] = self.signal[i]
            encoder_inputs[i, :, -1, :] = torch.zeros(1, 2)  # positional encoding of E
            decoder_inputs[i, :, 0, :] = torch.zeros(1, 2)  # positional encoding of S
            decoder_inputs[i, :, 1:, :] = self.signal[i]
            decoder_outputs[i, :, :-1, :] = self.signal[i]
            decoder_outputs[i, :, -1, :] = torch.zeros(1, 2)  # positional encoding of E
        return torch.Tensor(encoder_inputs), torch.Tensor(decoder_inputs), torch.Tensor(decoder_outputs)

    
class SignalDataset(Dataset):

    def __init__(self, encoder_inputs, decoder_inputs, decoder_outputs):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_outputs[idx]


if __name__ == '__main__':
    data_path = '../data/data_processed'
    reader1 = ReshapeDataset2('WN1', data_path, 'segmented', 400)
    trainset, _ = reader1()
    reader2 = ReshapeDataset2('WN2', data_path, 'segmented', 400)
    _, testset = reader2()
    a = Trainset2(trainset)
    encoder_inputs_train, decoder_inputs_train, decoder_outputs_train = a()
    b = Testset2(testset)
    encoder_inputs_test, decoder_inputs_test, decoder_outputs_test = b()
    data_loader = DataLoader(SignalDataset(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train),
                             batch_size=8,
                             shuffle=False
                             )
