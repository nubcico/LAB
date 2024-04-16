# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:12:51 2024

@author: madina.kudaibergenova

CSP feature extraction. OpenBCI 6 subjects.

"""
import numpy as np
from scipy.linalg import eigh
from Preprocess_OpenBCI_Data import OpenBCIPreprocessor

def csp_features():
    preprocessor_openbci = OpenBCIPreprocessor(subj_list=['sub01', 'sub02', 'sub14', 'sub17', 'sub19', 'sub22'], 
                                                path='M:/RA Spring 24/Transformer_openbci/dataset_openbci/', 
                                                band=[8, 12], interval=[150, 350])

    # Load and preprocess data
    openbci_raw_tr, openbci_raw_te, openbci_label_tr, openbci_label_te = preprocessor_openbci.load_and_preprocess_data()

    # Lists of 0-class and 1-class for train and test
    csp_tr0, csp_tr1, csp_te0, csp_te1 = [], [], [], []
    
    for tr_seg, tr_Label, te_seg, te_Label in zip(openbci_raw_tr, openbci_label_tr, openbci_raw_te, openbci_label_te):
        
        ##### train
        C1_tr = np.array(tr_seg)[np.where(tr_Label == 0)]
        C2_tr = np.array(tr_seg)[np.where(tr_Label == 1)]
        
        C1_tr = np.transpose(C1_tr, (2, 0, 1)) #50,20,200 -> 200,50,20
        C2_tr = np.transpose(C2_tr, (2, 0, 1)) #200,50,20
        C1_tr = np.reshape(C1_tr, (10000, 20)) #200*50,20
        C2_tr = np.reshape(C2_tr, (10000, 20)) #200*50,20
        
        Cov1_tr = np.cov(np.transpose(C1_tr))
        Cov2_tr = np.cov(np.transpose(C2_tr))
        
        D_tr, W_tr = eigh(Cov1_tr, Cov1_tr + Cov2_tr)
        CSP_W_tr = W_tr[:] 
    
        CSP_C1_tr = np.matmul(C1_tr, CSP_W_tr) # (200*50,20) x (20,20) -> (200*50,20)
        CSP_C2_tr = np.matmul(C2_tr, CSP_W_tr)
        
        CSP_C1_tr = np.reshape(CSP_C1_tr, (200, 50, 20), order='C')
        CSP_C1_tr = np.transpose(CSP_C1_tr, (1, 2, 0))
        CSP_C2_tr = np.reshape(CSP_C2_tr, (200, 50, 20), order='C')
        CSP_C2_tr = np.transpose(CSP_C2_tr, (1, 2, 0))
            
        csp_tr0.append(CSP_C1_tr)
        csp_tr1.append(CSP_C2_tr)
        
        ###### test
        C1_te = np.array(te_seg)[np.where(te_Label == 0)]
        C2_te = np.array(te_seg)[np.where(te_Label == 1)]
        
        C1_te = np.transpose(C1_te, (2, 0, 1))
        C2_te = np.transpose(C2_te, (2, 0, 1))
        C1_te = np.reshape(C1_te, (10000, 20))
        C2_te = np.reshape(C2_te, (10000, 20))
            
        CSP_C1_te = np.matmul(C1_te, CSP_W_tr)
        CSP_C2_te = np.matmul(C2_te, CSP_W_tr)
        
        CSP_C1_te = np.reshape(CSP_C1_te, (200, 50, 20), order='C')
        CSP_C1_te = np.transpose(CSP_C1_te, (1, 2, 0))
        CSP_C2_te = np.reshape(CSP_C2_te, (200, 50, 20), order='C')
        CSP_C2_te = np.transpose(CSP_C2_te, (1, 2, 0))
        
        csp_te0.append(CSP_C1_te)
        csp_te1.append(CSP_C2_te)
        
        
    # Concatenate CSP features for training data
    csp_tr_all, csp_tr_label_all = [], []
    for xtr0, xtr1 in zip(csp_tr0, csp_tr1):
        x_train = np.concatenate((xtr0, xtr1), axis=0)
        y_train = np.concatenate((np.zeros(50), np.ones(50)))
        csp_tr_all.append(x_train)
        csp_tr_label_all.append(y_train)
   
    # Concatenate CSP features for testing data
    csp_te_all, csp_te_label_all = [], []
    for xte0, xte1 in zip(csp_te0, csp_te1):
        x_test = np.concatenate((xte0, xte1), axis=0)
        y_test = np.concatenate((np.zeros(50), np.ones(50)))
        csp_te_all.append(x_test)
        csp_te_label_all.append(y_test)
        
    return csp_tr_all, csp_tr_label_all, csp_te_all, csp_te_label_all
