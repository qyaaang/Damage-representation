#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 4/05/21 9:05 PM
@description:  
@version: 1.0
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ReshapeDataset:

    def __init__(self, white_noise, data_path, data_source, len_seg):
        self.white_noise = white_noise
        self.data = np.load('{0}/{1}/{2}/{3}_{1}.npy'.format(data_path, data_source, len_seg, white_noise))
        self.len_seg = len_seg
        self.num_sensor = self.data.shape[0]
        self.num_channel = self.data.shape[1]
        self.num_seg = self.data.shape[2]
        self.num_feature = self.data.shape[3]

    def __call__(self, *args, **kwargs):
        print('Preparing {} dataset...'.format(self.white_noise))
        trainset, testset = self.reshape_trainset(), self.reshape_testset()
        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)
        trainset = torch.from_numpy(trainset)
        trainset = self.normalization(trainset)
        testset = torch.from_numpy(testset)
        for i in range(testset.size(0)):
            testset[i] = self.normalization(testset[i])
        return trainset, testset

    def reshape_trainset(self):
        """
        Reshape the trainset
        [14, 2, num_seg, num_feature] => [14 * num_seg, num_feature * 2]
        num_feature = length of each segments
        :return:
        """
        trainset = self.reshape_testset()
        # [14, num_seg, num_feature * 2] => [14 * num_seg, num_feature * 2]
        trainset = trainset.reshape((self.num_sensor * self.num_seg, -1))
        return trainset

    def reshape_testset(self):
        # [14, 2, num_seg, num_feature] => [14, num_seg, num_feature * 2]
        testset = np.zeros((self.num_sensor, self.num_seg, self.num_feature * self.num_channel))
        for i in range(self.num_sensor):
            tmp = self.data[i][0]
            for j in range(1, self.num_channel):
                tmp = np.hstack((tmp, self.data[i][j]))
            testset[i] = tmp
        return testset

    def decoder_input_init(self, dataset, last_time_step, is_trainset=True):
        """
        Last time step for the initial input of decoder
        :param dataset:
        :param last_time_step:
        :param is_trainset:
        """
        eos = np.zeros((self.num_sensor, self.num_seg))
        if last_time_step:
            if is_trainset:
                dataset = self.reshape_testset().astype(np.float32)
                dataset = torch.from_numpy(dataset)
                for i in range(dataset.size(0)):
                    dataset[i] = self.normalization(dataset[i])
            for i in range(self.num_sensor):
                for j in range(self.num_seg):
                    if j == 0:
                        eos[i, j] = 0
                    else:
                        eos[i, j] = dataset[i, j, self.len_seg - 1]
            return eos
        else:
            return eos

    @staticmethod
    def normalization(original_data, dim=1):
        """
        Normalize the signal to [-1, 1]
        :param original_data:
        :param dim:
        :return:
        """
        if dim == 1:
            d_min = torch.min(original_data, dim=dim)[0]
            for idx, j in enumerate(d_min):
                if j < 0:
                    original_data[idx, :] += torch.abs(d_min[idx])
                    d_min = torch.min(original_data, dim=dim)[0]
        else:
            d_min = torch.min(original_data, dim=dim)[0]
            for idx, j in enumerate(d_min):
                if j < 0:
                    original_data[idx, :] += torch.abs(d_min[idx])
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


class Trainset:

    def __init__(self, signal, trainset_eos, num_sensors=14):
        # [num_sensors * num_seg, len_seq] => [num_sensors * num_seg, len_seq, 1]
        self.signal = signal.unsqueeze(2)
        self.trainset_eos = trainset_eos
        self.num_sensors = num_sensors
        self.num_seg_all = signal.size(0)
        self.num_seg = int(self.num_seg_all / num_sensors)
        self.len_seq = signal.size(1)

    def __call__(self, *args, **kwargs):
        return self.encoding_trainset()

    def encoding_trainset(self):
        encoder_inputs = np.zeros((self.num_seg_all, self.len_seq + 1, 1))
        decoder_inputs = np.zeros((self.num_seg_all, self.len_seq + 1, 1))
        decoder_outputs = np.zeros((self.num_seg_all, self.len_seq + 1, 1))
        encoder_inputs[:, :-1, :] = self.signal
        encoder_inputs[:, -1, :] = 0  # positional encoding of E
        # Initial input of decoder (positional encoding of S)
        for i in range(self.num_sensors):
            decoder_inputs[i * self.num_seg: (i + 1) * self.num_seg, 0, :] = \
                self.trainset_eos[i].reshape((self.num_seg, 1))
        # decoder_inputs[:, 0, :] = 0  # positional encoding of S
        decoder_inputs[:, 1:, :] = self.signal
        decoder_outputs[:, :-1, :] = self.signal
        decoder_outputs[:, -1, :] = 0  # positional encoding of E
        return torch.Tensor(encoder_inputs), torch.Tensor(decoder_inputs), torch.Tensor(decoder_outputs)


class Testset:

    def __init__(self, signal, testset_eos):
        # [num_sensors, num_seg, len_seq] => [num_sensors, num_seg, len_seq, 1]
        self.signal = signal.unsqueeze(3)
        self.testset_eos = testset_eos
        self.num_sensors = signal.size(0)
        self.num_seg = signal.size(1)
        self.len_seq = signal.size(2)

    def __call__(self, *args, **kwargs):
        return self.encoding_testset()

    def encoding_testset(self):
        encoder_inputs = np.zeros((self.num_sensors, self.num_seg, self.len_seq + 1, 1))
        decoder_inputs = np.zeros((self.num_sensors, self.num_seg, self.len_seq + 1, 1))
        decoder_outputs = np.zeros((self.num_sensors, self.num_seg, self.len_seq + 1, 1))
        for i in range(self.num_sensors):
            encoder_inputs[i, :, :-1, :] = self.signal[i]
            encoder_inputs[i, :, -1, :] = 0  # positional encoding of E
            # Initial input of decoder (positional encoding of S)
            decoder_inputs[i, :, 0, :] = self.testset_eos[i].reshape((self.num_seg, 1))
            # decoder_inputs[i, :, 0, :] = 0  # positional encoding of S
            decoder_inputs[i, :, 1:, :] = self.signal[i]
            decoder_outputs[i, :, :-1, :] = self.signal[i]
            decoder_outputs[i, :, -1, :] = 0  # positional encoding of E
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
    reader1 = ReshapeDataset('WN1', data_path, 'segmented', 400)
    trainset, _ = reader1()
    reader2 = ReshapeDataset('WN2', data_path, 'segmented', 400)
    _, testset = reader2()
    trainset_eos = reader1.decoder_input_init(trainset, last_time_step=True, is_trainset=True)
    testset_eos = reader2.decoder_input_init(testset, last_time_step=True, is_trainset=False)
    a = Trainset(trainset, trainset_eos)
    encoder_inputs_train, decoder_inputs_train, decoder_outputs_train = a()
    b = Testset(testset, testset_eos)
    encoder_inputs_test, decoder_inputs_test, decoder_outputs_test = b()
    data_loader = DataLoader(SignalDataset(encoder_inputs_train, decoder_inputs_train, decoder_outputs_train),
                             batch_size=8,
                             shuffle=False
                             )
