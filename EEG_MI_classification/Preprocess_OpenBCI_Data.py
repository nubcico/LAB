# -*- coding: utf-8 -*-
"""
@author: madina.kudaibergenov

Preprocessing module for OpenBCI data.

"""

import scipy.io
import numpy as np
from scipy import signal
from scipy.signal import butter

class OpenBCIPreprocessor:
    """
    Class for preprocessing OpenBCI data.
    """
    def __init__(self, subj_list, path='C:/Users/madina.kudaibergenov/Dropbox/DATASETS/OpenBCI_dataset_6_subjects/',
                 band=[0.5, 40], interval=[150, 350]):
        """
        Initialize OpenBCIPreprocessor.

        Parameters:
        - subj_list (list): List of subject IDs
        - path (str): Path to the dataset directory.
        - band (list): Bandpass filter frequency range.
        - interval (list): Time interval for segmenting data.
        """
        self.path = path
        self.band = band
        self.interval = interval
        self.subj_list = subj_list
        self.CH = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6', 
                   'T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10',
                   'P7','P3','Pz','P4','P8','PO9''O1','Oz','O2','PO10','FC3','FC4',
                   'C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9','FTT9h',
                   'TTP7h','TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9',
                   'F10','AF7','AF3','AF4','AF8','PO3','PO4']
        self.ch_idx = self._select_channels()

    def _select_channels(self):
        """
        Select EEG channels.

        :return: ch_idx : List of selected channel indices.
        """
        ch_idx = []
        for i in self.CH:
            if 'C' in i:
                ch_idx.append(self.CH.index(i))
        return ch_idx

    def bandpass_filt(self, dat):    
        """
        Apply bandpass filter to data.

        :param: dat (numpy.ndarray): input data
        :return: fdat (numpy.ndarray): filtered data
        """
        sos = butter(5, self.band, 'band', fs=100, output='sos')   
        fdat = list()
        for i in range(np.size(dat, 1)):
            tm = signal.sosfilt(sos, dat[:,i])
            fdat.append(tm)  
        return np.array(fdat).transpose()
    
    def segment_data(self, cnt_data, marker, interval):
        """
        Segment the continuous data based on marker positions and interval.

        :param: cnt_data: Numpy array containing continuous data.
        :param: marker: Numpy array containing marker positions.
        :param: interval: List containing start and end indices of the desired segment.
        :return: seg_data: List of segmented data arrays.
        """
        seg_data = [cnt_data[range(marker[0, i] + interval[0], marker[0, i] + interval[1]), :] 
                    for i in range(np.size(marker, 1))]
        return seg_data
    
    def x_reshape(self, array):
        """
        Reshape input data.

        :param: array (numpy.ndarray): input data
        :return: array (numpy.ndarray): reshaped data into (trials,channels,samples)
        """
        return np.transpose(array, (0, 2, 1))

    def load_and_preprocess_data(self):
        """
        Load and preprocess OpenBCI data.

        :return: openbci_raw_tr (list): train data. filtered, segmented and reshaped
        :return: openbci_raw_te (list): test data. filtered, segmented and reshaped
        :return: openbci_label_tr (list):  train labels
        :return: openbci_label_te (list): test labels
        """
        openbci_raw_tr, openbci_raw_te, openbci_label_tr, openbci_label_te = [], [], [], []
        for subj in self.subj_list:
            # Load train data
            mat1 = scipy.io.loadmat(self.path + subj + 'tr_cnt.mat')
            tr_cnt = np.array(mat1.get('cnt1'))
            # Load train labels
            mat_Y1 = scipy.io.loadmat(self.path + subj + 'tr_label.mat')
            tr_Label = np.array(mat_Y1.get('label'))
            # Load train markers
            mat_m1 = scipy.io.loadmat(self.path + subj + 'tr_mrk.mat')
            tr_marker = np.array(mat_m1.get('marker'))
            tr_cnt = tr_cnt[:, self.ch_idx]
            tr_flt_cnt = self.bandpass_filt(tr_cnt)
            # Segment train data
            tr_seg = self.segment_data(tr_flt_cnt, tr_marker, self.interval)
            tr_seg = np.asarray(tr_seg)
            tr_seg = self.x_reshape(tr_seg)
            openbci_raw_tr.append(tr_seg)
            openbci_label_tr.append(tr_Label[0])
            
            # Load test data
            mat2 = scipy.io.loadmat(self.path + subj + 'te_cnt.mat')
            te_cnt = np.array(mat2.get('cnt1'))
            # Load test labels
            mat_Y2 = scipy.io.loadmat(self.path + subj + 'te_label.mat')
            te_Label = np.array(mat_Y2.get('label'))
            # Load test markers
            mat_m2 = scipy.io.loadmat(self.path + subj + 'te_mrk.mat')
            te_marker = np.array(mat_m2.get('marker'))
            te_cnt = te_cnt[:, self.ch_idx]
            te_flt_cnt = self.bandpass_filt(te_cnt)
            # Segment test data
            te_seg = self.segment_data(te_flt_cnt, te_marker, self.interval)
            te_seg = np.asarray(te_seg)
            te_seg = self.x_reshape(te_seg)
            openbci_raw_te.append(te_seg)
            openbci_label_te.append(te_Label[0])
            
        return openbci_raw_tr, openbci_raw_te, openbci_label_tr, openbci_label_te

    
if __name__ == '__main__':
    # Preprocessing OpenBCI dataset
    preprocessor_openbci = OpenBCIPreprocessor(subj_list=['sub01', 'sub02', 'sub14', 'sub17', 'sub19', 'sub22'], 
                                                path='C:/Users/madina.kudaibergenov/Dropbox/DATASETS/OpenBCI_dataset_6_subjects/',
                                                band=[0.5, 40], interval=[150, 350])
    
    # Preprocess OpenBCI dataset
    openbci_train, openbci_test, openbci_label_train, openbci_label_test = preprocessor_openbci.load_and_preprocess_data()


