# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:43:21 2024

@author: madina.kudaibergenova

Example usage

"""

from Preprocess_BCIIV2a_Data import BCIIV2aPreprocessor

preprocessor = BCIIV2aPreprocessor(subj_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9], x_path='M:/RA Spring 24/Baseline_codes/BCICIV_2a_gdf/', 
             y_path='M:/RA Spring 24/Baseline_codes/true_labels/', band=[0.5, 40])

#4 classes
x_train, x_test, y_train, y_test = preprocessor.preprocess_4class_dataset()
#2 classes
x_train_2cl, x_test_2cl, y_train_2cl, y_test_2cl = preprocessor.preprocess_2class_dataset()