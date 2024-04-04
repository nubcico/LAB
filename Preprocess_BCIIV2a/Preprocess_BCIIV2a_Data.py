# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:24:01 2024

@author: madina.kudaibergenova

BCI IV 2a dataset preprocessing: loading, filtering, segmentation.

"""
import scipy.io
import numpy as np
import os
import mne
from scipy.signal import butter, sosfilt

class BCIIV2aPreprocessor:
    def __init__(self, subj_ids, x_path='M:/RA Spring 24/Baseline_codes/BCICIV_2a_gdf/', 
                 y_path='M:/RA Spring 24/Baseline_codes/true_labels/', 
                 band=[0.5, 40]):
        self.subj_ids = subj_ids
        self.x_path = x_path
        self.y_path = y_path
        self.band = band
        self.CH = ['EEG-Fz','EEG-0','EEG-1','EEG-2','EEG-3','EEG-4','EEG-5',
                   'EEG-C3','EEG-6','EEG-Cz','EEG-7','EEG-C4','EEG-8','EEG-9',
                   'EEG-10','EEG-11','EEG-12','EEG-13','EEG-14','EEG-Pz','EEG-15',
                   'EEG-16','EOG-left','EOG-central','EOG-right']
        self.ch_idx = self._select_channels()

    def _select_channels(self):
        # Function to select EEG channels
        eeg_ch_idx = [i for i, ch_name in enumerate(self.CH) if 'EEG' in ch_name]
        return eeg_ch_idx

    def load_true_labels_4class(self):
        # Load true labels for subjects
        train_labels_all, test_labels_all = [], []    
        for subj_id in self.subj_ids:
            tr_label_mat = scipy.io.loadmat(os.path.join(self.y_path, f'A0{subj_id}T.mat'))
            te_label_mat = scipy.io.loadmat(os.path.join(self.y_path, f'A0{subj_id}E.mat'))
            tr_label_arr = np.array(tr_label_mat.get('classlabel'))
            te_label_arr = np.array(te_label_mat.get('classlabel'))
            tr_label_arr_zero = tr_label_arr - 1  # Convert to 0-based true labels
            te_label_arr_zero = te_label_arr - 1
            train_labels_all.append(tr_label_arr_zero)
            test_labels_all.append(te_label_arr_zero)

        return train_labels_all, test_labels_all

    def extract_true_labels_2class(self):
        train_labels_all, test_labels_all = self.load_true_labels_4class()
        tr_label_bin_list, te_label_bin_list = [], []
        # Extract only 2 classes from the labels
        tr_label_bin_list = [bin_label[(bin_label == 0) | (bin_label == 1)] for bin_label in train_labels_all]
        te_label_bin_list = [bin_label[(bin_label == 0) | (bin_label == 1)] for bin_label in test_labels_all]
        return tr_label_bin_list, te_label_bin_list

    def load_dataset_gdf(self):
        # Load EEG data for subjects
        bciiv2a_tr_list, bciiv2a_te_list = [],[]
        for subj_id in self.subj_ids:
            tr_gdf_file_name = f"A0{subj_id}T.gdf"
            te_gdf_file_name = f"A0{subj_id}E.gdf"
            full_tr_path = os.path.join(self.x_path, tr_gdf_file_name)
            full_te_path = os.path.join(self.x_path, te_gdf_file_name)
            
            # Read raw data using MNE toolbox
            tr_raw_2a_gdf = mne.io.read_raw_gdf(full_tr_path, preload=True)
            te_raw_2a_gdf = mne.io.read_raw_gdf(full_te_path, preload=True)
            
            # Append raw data to train and test lists
            bciiv2a_tr_list.append(tr_raw_2a_gdf)
            bciiv2a_te_list.append(te_raw_2a_gdf)
            
        return bciiv2a_tr_list, bciiv2a_te_list

    def select_eeg_chans(self):
        # Select EEG channels
        eeg_ch_name = [self.CH[idx] for idx in self.ch_idx]
        return eeg_ch_name, self.ch_idx

    def arr_from_raw(self, list_l):
        # Convert raw data to numpy arrays
        return [l.get_data() for l in list_l]

    def select_ch_raw_arr(self, list_l):
        # Select EEG channels from raw data
        return [l[self.ch_idx, :] for l in list_l]

    def bandpass_filter(self, list_l):   
        # Apply bandpass filtering
        sos = butter(5, self.band, 'bandpass', fs=250, output='sos')  
        return [np.array([sosfilt(sos, ch_data) for ch_data in subj_data]) for subj_data in list_l]

    def segment_data_upd(self, list_l, list_t, ival1, ival2):
        # Segment the data
        seg_list = []
        for l, t in zip(list_l, list_t):
            l_list_sub = []
            for i in range(len(t)):
                interval = range(int(t[i]) + int(ival1), int(t[i]) + int(ival2))
                l_list_sub.append(l[:, interval].T)
            seg_list.append(l_list_sub)
        return seg_list

    def create_x_data(self, list_l):
        # Create X data from continuous data
        x_list = [np.array([trial_data for trial_data in subj]) for subj in list_l]
        return x_list

    def x_reshape(self, list_l, input_shape):
        # Reshape X data
        return [np.transpose(l, (0, 2, 1)).reshape(l.shape[0], *input_shape) for l in list_l]
    
    def extract_event_sec_given_cue(self, list_l, cue_id, chans):
        event_cue_id = {f'{cue_id}': cue_id}
        list_event_sec = []
        for l in list_l:
            l = l.pick_channels(chans)
            events, event_id = mne.events_from_annotations(l, event_id = event_cue_id)
            event_sec = [int(event[0]) for event in events] #1st column is time of events in seconds
            list_event_sec.append(event_sec)
        return list_event_sec   

    def extract_2_classes(self, list_l, channels):
        # Extract data for two classes
        eve_list_2_class, eve_sec_list_2_class, event_id_list_2_class, l_list_raw = [], [], [], []
        event_id_2_class = {'769': 0, '770': 1}  # Mapping event IDs for two classes
        for l in list_l:
            l = l.pick_channels(channels)
            events_2_class, _ = mne.events_from_annotations(l, event_id=event_id_2_class)
            l_list_raw.append(l)
            eve_list_2_class.append(events_2_class)
            eve_sec_2_class = [int(event[0]) for event in events_2_class]
            eve_sec_list_2_class.append(eve_sec_2_class)
            eve_id_2_class = [int(event_id[2]) for event_id in events_2_class]
            eve_id_map_2_class = {event_id: i for i, event_id in enumerate(sorted(set(eve_id_2_class)))}
            eve_id_2_class_rename = [eve_id_map_2_class[event_id] for event_id in eve_id_2_class]
            event_id_list_2_class.append(np.array(eve_id_2_class_rename))
        return eve_list_2_class, eve_sec_list_2_class, event_id_list_2_class, l_list_raw


### just add one more def and return train/test/labels
    def preprocess_4class_dataset(self):
        y_train, y_test = self.load_true_labels_4class()
        train_data, test_data = self.load_dataset_gdf()
        channels, _ = self.select_eeg_chans()
        xtrain = self.arr_from_raw(train_data)
        xtrain = self.select_ch_raw_arr(xtrain)
        xtest = self.arr_from_raw(test_data)
        xtest = self.select_ch_raw_arr(xtest)
        x_train_filtered = self.bandpass_filter(xtrain)
        x_test_filtered = self.bandpass_filter(xtest)
        
        train_sec_4cl, test_sec_4cl = [], []
        train_sec_4cl = self.extract_event_sec_given_cue(train_data, 768, channels)
        test_sec_4cl = self.extract_event_sec_given_cue(test_data, 768, channels)

        ### segment filtered dataset with the above interval, as it was from fix point, therefore +[3.5*250=875] and +[5.5*250=1375]
        train_flt_seg = self.segment_data_upd(x_train_filtered, train_sec_4cl, 875, 1375)
        test_flt_seg = self.segment_data_upd(x_test_filtered, test_sec_4cl, 875, 1375)
        train_flt_seg = self.create_x_data(train_flt_seg)
        test_flt_seg = self.create_x_data(test_flt_seg)
        
        Chans = 22  # Number of EEG channels
        Samples = 500  # Number of samples
        input_shape = (Chans, Samples)

        train_flt_seg_reshaped = self.x_reshape(train_flt_seg, input_shape)
        test_flt_seg_reshaped = self.x_reshape(test_flt_seg, input_shape)
        
        #just renaming, for convenience
        x_train = train_flt_seg_reshaped
        x_test = test_flt_seg_reshaped
        
        return x_train, x_test, y_train, y_test
    

    def preprocess_2class_dataset(self):
        
        y_train, y_test = self.extract_true_labels_2class()
        train_data, test_data = self.load_dataset_gdf()
        channels, _ = self.select_eeg_chans()
        xtrain = self.arr_from_raw(train_data)
        xtrain = self.select_ch_raw_arr(xtrain)
        xtest = self.arr_from_raw(test_data)
        xtest = self.select_ch_raw_arr(xtest)
        x_train_filtered = self.bandpass_filter(xtrain)
        x_test_filtered = self.bandpass_filter(xtest)
        
        # as there is no labels for test data, unknown cue will be extracted    
        test_eve_unk_cue = self.extract_event_sec_given_cue(test_data, 783, channels)
        # Filter events time for labels 0 and 1
        test_event_time_2cl = []
        for t, l in zip(test_eve_unk_cue, test_data):
            events_time_01 = [t[i] for i, label in enumerate(l) if label == 0 or label == 1]
            test_event_time_2cl.append(events_time_01) #list has exact time points of 0 and 1 classes for test to we can segment it
            
        #eve_list_2_class, eve_sec_list_2_class, event_id_list_2_class, l_list_raw  #3columns, seconds, y_labels, raw_gdf
        _, train_event_time_2cl, _, _ = self.extract_2_classes(train_data, channels)

        ### segment filtered dataset with the above interval
        train_flt_seg_2cl = self.segment_data_upd(x_train_filtered, train_event_time_2cl, 375, 875)
        test_flt_seg_2cl = self.segment_data_upd(x_test_filtered, test_event_time_2cl, 375, 875)
        train_flt_xdata_2cl = self.create_x_data(train_flt_seg_2cl)
        test_flt_xdata_2cl = self.create_x_data(test_flt_seg_2cl)    

        Chans = 22  # Number of EEG channels
        Samples = 500  # Number of samples
        input_shape = (Chans, Samples)

        train_flt_xdata_2cl_res = self.x_reshape(train_flt_xdata_2cl, input_shape)
        test_flt_xdata_2cl_res = self.x_reshape(test_flt_xdata_2cl, input_shape)
        
        #just renaming, for convenience
        x_train = train_flt_xdata_2cl_res
        x_test = test_flt_xdata_2cl_res
        
        return x_train, x_test, y_train, y_test

          
    
    