# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:48:13 2024

@author: madina.kudaibergenova

Preprocessing downsampled data of OpenBCI dataset, all 54 subjects.

"""

import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt

class OpenBCI_Preprocess_Dataset:
    def __init__(self, path):
        self.path = path
        self.band = [0.5, 40]

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=0)
        return y

    def bandpass_filter(self, data):
        for trial in range(data.shape[1]):  # Iterate over trials
            for channel in range(data.shape[2]):  # Iterate over channels
                data[:, trial, channel] = self.butter_bandpass_filter(data[:, trial, channel], self.band[0], self.band[1], fs=100)
        return data

    def preprocess_data(self):
        openbci_train, openbci_test, openbci_label_train, openbci_label_test = [], [], [], []
        
        for i in range(1, 55):  # Loop from 1 to 54
            subject_number = f's{i:02}'
            # Load train data
            mat1 = scipy.io.loadmat(self.path + f'{subject_number}/train_x_downsampled.mat')
            tr_x = mat1['smt_train_downsampled']

            # Load train labels
            mat_Y1 = scipy.io.loadmat(self.path + f'{subject_number}/train_y.mat')
            tr_Label = mat_Y1['y_logic_train']

            tr_x_flt = self.bandpass_filter(tr_x)
            tr_x_flt = np.transpose(tr_x_flt, (1, 2, 0))

            openbci_train.append(tr_x_flt)
            openbci_label_train.append(tr_Label[0])

            # Load test data
            mat2 = scipy.io.loadmat(self.path + f'{subject_number}/test_x_downsampled.mat')
            te_x = mat2['smt_test_downsampled']

            # Load test labels
            mat_Y2 = scipy.io.loadmat(self.path + f'{subject_number}/test_y.mat')
            te_Label = mat_Y2['y_logic_test']

            te_x_flt = self.bandpass_filter(te_x)
            te_x_flt = np.transpose(te_x_flt, (1, 2, 0))

            openbci_test.append(te_x_flt)
            openbci_label_test.append(te_Label[0])

        return openbci_train, openbci_test, openbci_label_train, openbci_label_test

if __name__ == '__main__':
    # Initialize preprocessor
    session1 = 'C:/Users/madina.kudaibergenov/Dropbox/DATASETS/OpenBMI_Dataset_Motor_Imagery/downsampled_dataset/session1/'
    session2 = 'C:/Users/madina.kudaibergenov/Dropbox/DATASETS/OpenBMI_Dataset_Motor_Imagery/downsampled_dataset/session2/'
    preprocessor_session1 = OpenBCI_Preprocess_Dataset(session1)
    
    # Preprocess data
    openbci_train, openbci_test, openbci_label_train, openbci_label_test = preprocessor_session1.preprocess_data()

