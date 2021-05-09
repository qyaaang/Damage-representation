#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Qun Yang
@license: (C) Copyright 2021, University of Auckland
@contact: qyan327@aucklanduni.ac.nz
@date: 4/05/21 10:38 AM
@description:  
@version: 1.0
"""


import numpy as np
import pandas as pd
import json
import os
import argparse
import sys
import data_processing as dp

if sys.platform == 'darwin':
    data_path = '/Users/qunyang/Dropbox (Personal)/Dataset/DesignSafe-CI/UCSD'
else:
    data_path = r'H:\EngFiles\Dataset\UCSD'
with open('./data/info/folders.json') as f:
    folders = json.load(f)
info_path = './data/info'
white_noises = ['WN2']


class Preprocessing:

    def __init__(self, args):
        self.args = args
        self.segmented_path = './data/data_preprocessed/segmented data'
        self.denoised_path = './data/data_preprocessed/denoised data'
        self.fft_path = './data/data_preprocessed/FFT'

    def __call__(self, *args, **kwargs):
        self.seg_signal()
        self.denoise()
        # self.fft()

    def seg_signal(self):
        """
        Segment signal
        """
        num_segs = {}
        sensors = np.load('{}/sensors.npy'.format(info_path))  # Sensors
        for white_noise in white_noises:
            data = pd.DataFrame()
            signal = dp.Segmentation(data_path, folders, white_noise, self.args.len_seg)
            for sensor_name in sensors:
                data_split, num_seg = signal(sensor_name)
                data = pd.concat([data, data_split], axis=1)
            num_segs[white_noise] = num_seg
            print('Signal segmentation for {} completed.'.format(white_noise))
            if not os.path.exists('{}/{}'.format(self.segmented_path, self.args.len_seg)):
                os.mkdir('{}/{}'.format(self.segmented_path, self.args.len_seg))
            data.to_csv('{}/{}/{}_segmented.csv'.
                        format(self.segmented_path, self.args.len_seg, white_noise), index=None)
        num_segs = json.dumps(num_segs, indent=2)
        with open('./data/info/segments_{}.json'.format(self.args.len_seg), 'w') as f:
            f.write(num_segs)

    def denoise(self):
        """
        Denoise signal with wavelet transformation
        """
        for white_noise in white_noises:
            signal = dp.Denoise(self.segmented_path, white_noise, self.args.len_seg)
            data = signal()
            print('Signal denoise for {} completed.'.format(white_noise))
            if not os.path.exists('{}/{}'.format(self.denoised_path, self.args.len_seg)):
                os.mkdir('{}/{}'.format(self.denoised_path, self.args.len_seg))
            data.to_csv('{}/{}/{}_denoised.csv'.
                        format(self.denoised_path, self.args.len_seg, white_noise), index=None)

    def fft(self):
        """
        FFT
        """
        for white_noise in white_noises:
            signal = dp.FFT(self.denoised_path, white_noise, self.args.len_seg)
            data = signal()
            print('Signal FFT for {} completed.'.format(white_noise))
            if not os.path.exists('{}/{}'.format(self.fft_path, self.args.len_seg)):
                os.mkdir('{}/{}'.format(self.fft_path, self.args.len_seg))
            data.to_csv('{}/{}/{}_FFT.csv'.
                        format(self.fft_path, self.args.len_seg, white_noise), index=None)


class Processing:

    def __init__(self, args):
        self.args = args
        self.segmented_path = './data/data_preprocessed/segmented data'
        self.denoised_path = './data/data_preprocessed/denoised data'
        self.fft_path = './data/data_preprocessed/FFT'
        self.segmented_save_path = './data/data_processed/segmented'
        self.denoised_save_path = './data/data_processed/denoised'
        self.fft_save_path = './data/data_processed/FFT'
        self.sources = {'1': {'Path': self.segmented_path,
                              'Save path': self.segmented_save_path,
                              'Type': 'segmented',
                              'Dim': args.len_seg
                              },
                        '2': {'Path': self.denoised_path,
                              'Save path': self.denoised_save_path,
                              'Type': 'denoised',
                              'Dim': args.len_seg
                              },
                        '3': {'Path': self.fft_path,
                              'Save path': self.fft_save_path,
                              'Type': 'FFT',
                              'Dim': 120
                              }
                        }

    def __call__(self, *args, **kwargs):
        self.write_array()

    def write_array(self):
        """

        """
        spots = np.load('{}/spots.npy'.format(info_path))  # Sensor spots
        dirs = ['N', 'E']  # Sensor directions
        with open('{}/segments_{}.json'.format(info_path, self.args.len_seg)) as f:
            segs = json.load(f)
        for white_noise in white_noises:
            data = pd.read_csv('{}/{}/{}_{}.csv'.
                               format(self.sources[self.args.data_source]['Path'],
                                      self.args.len_seg,
                                      white_noise,
                                      self.sources[self.args.data_source]['Type']
                                      )
                               )
            w = np.zeros((len(spots),
                          len(dirs),
                          segs[white_noise],
                          self.sources[self.args.data_source]['Dim']
                          )
                         )
            for idx, spot in enumerate(spots):
                level, location = spot.split('-')[0], spot.split('-')[1]
                for i, d in enumerate(dirs):
                    for seg in range(segs[white_noise]):
                        w[idx][i][seg][:] = data['AC{}{}{}_{}'.format(level, d, location, seg + 1)]
            if not os.path.exists('{}/{}'.format(self.sources[self.args.data_source]['Save path'],
                                                 self.args.len_seg
                                                 )):
                os.mkdir('{}/{}'.format(self.sources[self.args.data_source]['Save path'],
                                        self.args.len_seg
                                        )
                         )
            np.save('{}/{}/{}_{}.npy'.
                    format(self.sources[self.args.data_source]['Save path'],
                           self.args.len_seg,
                           white_noise,
                           self.sources[self.args.data_source]['Type']
                           ),
                    w
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--len_seg', default=400, type=int)
    parser.add_argument('--data_source', default='1', type=str)
    parser.add_argument('--mode', default='0', type=str)
    opts = parser.parse_args()
    Prep = Preprocessing(opts)  # Preprocessing
    Pro = Processing(opts)  # Processing
    Prep() if opts.mode == '0' else Pro()
