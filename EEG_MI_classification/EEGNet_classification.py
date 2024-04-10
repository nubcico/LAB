# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:39:45 2024

@author: madina.kudaibergenova

EEGNet

"""
import numpy as np
from Preprocess_OpenBCI_Data import OpenBCIPreprocessor
from CSP_main import csp_features
from Transformer_model_main import EEGDataset, ModelWrapper, EEGClassificationModel, initialize_trainer
from EEG_models import EEGNet
from tensorflow.keras.optimizers import Adam
import time
timestamp = time.strftime("%Y%m%d_%H%M%S")

class OpenBCIClassification:
    def __init__(self, subj_list, path, band, interval, checkpoint_dir):
        self.subj_list = subj_list
        self.path = path
        self.band = band
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir

    def preprocess_data(self):
        preprocessor_openbci = OpenBCIPreprocessor(subj_list=self.subj_list, 
                                                    path=self.path, 
                                                    band=self.band, 
                                                    interval=self.interval)
        openbci_train, openbci_test, openbci_label_train, openbci_label_test = preprocessor_openbci.load_and_preprocess_data()
        return openbci_train, openbci_test, openbci_label_train, openbci_label_test

    def load_csp_features(self):
        return csp_features()

    def initialize_datasets_and_models(self, x_tr_list, x_te_list, tr_label_list, te_label_list):
        eeg_dataset_list = []
        for x_tr, x_te, y_tr, y_te in zip(x_tr_list, x_te_list, tr_label_list, te_label_list):
            eeg_dataset = EEGDataset(x_tr=x_tr, x_te=x_te, y_tr=y_tr, y_te=y_te)
            eeg_dataset_list.append(eeg_dataset)

        model_list = []
        for eeg_data in eeg_dataset_list:
            model = EEGClassificationModel(eeg_channel=20, dropout=0.125)
            model_wrap = ModelWrapper(model, eeg_data, batch_size=10, lr=5e-4, max_epoch=100)
            model_list.append(model_wrap)
        return model_list

    def train_and_test_models(self, model_list):
        trainer_list = []
        for m in model_list: 
            trainer = initialize_trainer(SEED=int(np.random.randint(2147483647)), CHECKPOINT_DIR=self.checkpoint_dir, MAX_EPOCH=100)
            trainer.fit(m)
            trainer_list.append(trainer)

        test_res_list = []
        for m, t in zip(model_list, trainer_list):
            test_res = t.test(m)
            test_res_list.append(test_res)

        test_res_acc_list = []
        for i in test_res_list:
            for j in i:
                print(j['test_acc'])
                test_res_acc_list.append(j['test_acc'])

        return test_res_acc_list, np.average(test_res_acc_list)
    
    def EEGNet_classification_2cl(self, x_train, x_test, y_train, y_test, checkpoint_dir):
        EEGNet_loss_list_2cl, EEGNet_acc_list_2cl = [], []
        for i,(x_tr, x_te, y_tr, y_te) in enumerate(zip(x_train, x_test, y_train, y_test)):
        
            model = EEGNet(nb_classes=2, Chans=20, Samples=200)
            model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
            # Train the model
            model.fit(x_tr, y_tr, epochs=200, batch_size=32, validation_data = (x_te, y_te))
            model.save(checkpoint_dir + f'model_eegnet_2cl_subj{i}_{timestamp}.h5')
            
            # Evaluate the loaded model on the test data
            loss, accuracy = model.evaluate(x_te, y_te)
            EEGNet_loss_list_2cl.append(loss)
            EEGNet_acc_list_2cl.append(accuracy)
        
            print(f'Test Loss: {loss:.4f}')
            print(f'Test Accuracy: {accuracy:.4f}')
        
        return EEGNet_acc_list_2cl

if __name__ == "__main__":

    #checkpoint_dir = 'C:/Users/madina.kudaibergenov/Dropbox/Lab_members/Madina/Codes/OpenBCI/models_results_etc/'
    checkpoint_dir = 'M:/RA Spring 24/Transformer_openbci/results/'

    openbci_classifier = OpenBCIClassification(subj_list=['sub01', 'sub02', 'sub14', 'sub17', 'sub19', 'sub22'], 
                                             path='C:/Users/madina.kudaibergenov/Dropbox/DATASETS/OpenBCI_dataset_6_subjects/', 
                                             band=[0.5, 40], interval=[150, 350], checkpoint_dir=checkpoint_dir)

    ############## Preprocess Data ##############
    openbci_train, openbci_test, openbci_label_train, openbci_label_test = openbci_classifier.preprocess_data()

    ############## Load CSP features ##############
    csp_tr_all, csp_tr_label, csp_te_all, csp_te_label = openbci_classifier.load_csp_features()
    
    ############## EEGNet subject dependent individual classification ##############
    EEGNet_acc_list_2cl = openbci_classifier.EEGNet_classification_2cl(openbci_train, openbci_test, openbci_label_train, openbci_label_test, checkpoint_dir)
    print("EEGNet subject dependent results:")
    for i in range(len(EEGNet_acc_list_2cl)):
         print(f'Subject {i}: ', EEGNet_acc_list_2cl[i])
    print("EEGNet average: ", np.average(EEGNet_acc_list_2cl))
    
    ##############Transformer subject-dependent individual training ##############
    openbci_sd_model_list = openbci_classifier.initialize_datasets_and_models(openbci_train, openbci_test, openbci_label_train, openbci_label_test)
    openbci_sd_test_res, openbci_sd_test_res_ave = openbci_classifier.train_and_test_models(openbci_sd_model_list)
    print("Average subject-dependent accuracy:", openbci_sd_test_res_ave)

    ############## Subject-dependent + CSP individual training ##############
    openbci_sd_csp_model_list = openbci_classifier.initialize_datasets_and_models(csp_tr_all, csp_te_all, csp_tr_label, csp_te_label)
    openbci_sd_csp_test_res, openbci_sd_csp_test_res_ave = openbci_classifier.train_and_test_models(openbci_sd_csp_model_list)
    print("Average subject-dependent using CSP accuracy:", openbci_sd_csp_test_res_ave)
    
    
    