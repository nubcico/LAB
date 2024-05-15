# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:09:03 2024

@author: madina.kudaibergenova

OpenBCI 54 subjects. EEGNet, DeepConvNet, ShallowConvNet, Transformer. Subject independent without fine-tuning the models. LOSO-CV approach.
"""

import numpy as np
from Preprocess_OpenBCI_AllData import OpenBCI_Preprocess_Dataset
from Transformer_model_main import EEGDataset_LOSO_CV, ModelWrapper, EEGClassificationModel, initialize_trainer
from EEG_models import EEGNet, DeepConvNet, ShallowConvNet, EEGNet_mt
from tensorflow.keras.optimizers import Adam
import time
timestamp = time.strftime("%Y%m%d_%H%M%S")
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os
import tensorflow as tf

# load datasets
class DataPreprocessor:
    def __init__(self, session_path, save_path_var):
        self.session_path = session_path
        self.save_path_var = save_path_var

    def preprocess_data(self):
        preprocessor_session = OpenBCI_Preprocess_Dataset(self.session_path)
        # Preprocess data
        openbci_train, openbci_test, openbci_label_train, openbci_label_test = preprocessor_session.preprocess_data()
        return openbci_train, openbci_test, openbci_label_train, openbci_label_test

    def init_dataset(self, openbci_train_pkl, openbci_test_pkl, openbci_label_train_pkl, openbci_label_test_pkl):
        datasets = []
        for subj in range(54):
            dataset = EEGDataset_LOSO_CV(openbci_train_pkl, openbci_test_pkl, openbci_label_train_pkl,
                                         openbci_label_test_pkl, subj)
            datasets.append(dataset)
        return datasets

    def save_variable(self, var_name, variable):
        with open(self.save_path_var + f'{var_name}.pkl', 'wb') as f:
            pickle.dump(variable, f)

    def read_variable(self, var_name):
        with open(self.save_path_var + f'{var_name}.pkl', 'rb') as f:
            variable = pickle.load(f)
        return variable

    def save_pkl_dataset(self):
        # Preprocess data
        openbci_train, openbci_test, openbci_label_train, openbci_label_test = self.preprocess_data()
        # Save datasets
        self.save_variable('openbci_train', openbci_train)
        self.save_variable('openbci_test', openbci_test)
        self.save_variable('openbci_label_train', openbci_label_train)
        self.save_variable('openbci_label_test', openbci_label_test)

    def read_pkl_dataset(self):
        # Read saved datasets
        openbci_train_pkl = self.read_variable('openbci_train')
        openbci_test_pkl = self.read_variable('openbci_test')
        openbci_label_train_pkl = self.read_variable('openbci_label_train')
        openbci_label_test_pkl = self.read_variable('openbci_label_test')

        return openbci_train_pkl, openbci_test_pkl, openbci_label_train_pkl, openbci_label_test_pkl

    def save_eeg_dataset(self):
        openbci_train_pkl, openbci_test_pkl, openbci_label_train_pkl, openbci_label_test_pkl = self.read_pkl_dataset()
        # Initialize datasets from saved files
        datasets = self.init_dataset(openbci_train_pkl, openbci_test_pkl, openbci_label_train_pkl,
                                     openbci_label_test_pkl)
        # Save datasets list
        self.save_variable('datasets', datasets)

    def read_eeg_dataset(self):
        # Read datasets list
        datasets = self.read_variable('datasets')
        return datasets
              
# transformer    
class TransformerClassification():
    def __init__(self, checkpoint_dir_vit):
        self.CLASSES = [0, 1]
        self.MODEL_NAME = "EEGClassificationModel"
        self.MAX_EPOCH = 1
        self.BATCH_SIZE = 64
        self.LR = 5e-4
        self.CHECKPOINT_DIR = checkpoint_dir_vit
        self.SEED = int(np.random.randint(2147483647))
        self.EEG_CHANNEL = 20
        self.SAMPLES = 250

    # initialize eeg models
    def init_models(self, datasets):
        model_list = []
        for eeg_data in datasets:
            model = EEGClassificationModel(eeg_channel=self.EEG_CHANNEL, dropout=0.125)
            model_wrap = ModelWrapper(model, eeg_data, self.BATCH_SIZE, self.LR, self.MAX_EPOCH)
            model_list.append(model_wrap)
        return model_list
    
    #result will print all results and return the list as well as return average result
    def train_and_test_models(self, model_list):
        """Train and test models. Return results."""
        trainer_list = []
        for m in model_list:
            trainer = initialize_trainer(self.SEED, self.CHECKPOINT_DIR, self.MAX_EPOCH)
            trainer.fit(m)
            trainer_list.append(trainer)
    
        #will store loss and test acc results
        test_res_list = []
        for m, t in zip(model_list, trainer_list):
            test_res = t.test(m)
            test_res_list.append(test_res)
    
        #extract only accuracy results
        test_res_acc_list = []
        for i in test_res_list:
            for j in i:
                print(j['test_acc'])
                test_res_acc_list.append(j['test_acc'])
    
        return test_res_acc_list, np.average(test_res_acc_list)

# EEGNet, DeepConvNet, ShallowConvNet single loss
class CNNsOneLossClassification():
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.Chans=20
        self.Samples=250
        self.epochs=1
        self.batch_size=64
        self.num_classes = 2

    # initialize CNN models
    def train_model_classification(self, model_type, x_train, x_test, y_train, y_test):
        model_loss_list, model_acc_list = [], []
        model_saved = []
        for i, (x_tr, x_te, y_tr, y_te) in enumerate(zip(x_train, x_test, y_train, y_test)):
            model = model_type(nb_classes=self.num_classes, Chans=self.Chans, Samples=self.Samples)
            model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            # Train the model
            model.fit(x_tr, y_tr, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_te, y_te))
            model_saved.append(model)
            model.save(self.checkpoint_dir + f'model_{model_type.__name__}_subj{i}_{timestamp}.h5')
            # Evaluate the loaded model on the test data
            loss, accuracy = model.evaluate(x_te, y_te)
            model_loss_list.append(loss)
            model_acc_list.append(accuracy)
            print(f'Test Loss: {loss:.4f}')
            print(f'Test Accuracy: {accuracy:.4f}')
            tf.keras.backend.clear_session()
        return model_acc_list, model_saved

# EEGNet two losses
class EEGNetTwoLosses():
    def __init__(self, save_path, y_train_all, y_test_all):
        self.num_subjects = 54
        self.num_classes = 2
        self.Chans = 20
        self.Samples = 250
        self.epochs = 1
        self.batch_size = 64
        
        self.losses = {
            "softmax_subject": "categorical_crossentropy",  # Subject identification loss
            "dense_lr": "categorical_crossentropy"  # Motor imagery classification loss (for now categorical as we use [100,2] labels)
        }
        self.loss_weights = {"softmax_subject": 0.1, "dense_lr": 0.9}  # weights for both losses
        self.optimizer = Adam()
        
        self.one_hot_labels_test = self.create_onehot_labels_test(54, 100)
        self.y_train_all_encoded, self.y_test_all_encoded = self.make_labels_categorical(y_train_all, y_test_all)
        
        self.save_path = save_path
        
    def create_onehot_labels_test(self, num_subjects, num_trials):
        num_subjects = num_subjects
        num_trials = num_trials
        # Generate labels for each subject
        labels = np.repeat(np.arange(num_subjects), num_trials)
        # Convert labels to one-hot encoded format
        one_hot_labels = to_categorical(labels, num_subjects)
        # Split one-hot encoded labels into a list of 54 subjects
        one_hot_labels_test = [one_hot_labels[i*num_trials:(i+1)*num_trials] for i in range(num_subjects)]
        return one_hot_labels_test

    def create_onehot_labels_train(self):
        # Adjust the size of each subject's labels to match the number of samples in x_train_all
        one_hot_labels_train = []
        for labels_subject in self.one_hot_labels_test:
            # Repeat each label 106 times to match the number of samples in x_train_all
            labels_subject_train = np.repeat(labels_subject, 106, axis=0)
            one_hot_labels_train.append(labels_subject_train)
        return one_hot_labels_train
    
    def make_labels_categorical(self, y_train_all, y_test_all):
        # this is temporarly created additional method to help me with the two losses classification
        y_train_all_encoded = [to_categorical(label, 2) for label in y_train_all]
        y_test_all_encoded = [to_categorical(label, 2) for label in y_test_all]
        return y_train_all_encoded, y_test_all_encoded

    def model_training(self, x_train_all, x_test_all):
        history_all = []
        evaluation_all = []
        one_hot_labels_train = self.create_onehot_labels_train()
        for i, (x_tr, x_te, y_tr1, y_te1, y_tr2, y_te2) in enumerate(zip(x_train_all, x_test_all, 
                                                                         one_hot_labels_train, self.one_hot_labels_test, 
                                                                         self.y_train_all_encoded, self.y_test_all_encoded)):
            model_mt = EEGNet_mt(nb_subjects=self.num_subjects, nb_classes=self.num_classes, Chans=self.Chans, Samples=self.Samples)
            model_mt.compile(optimizer=self.optimizer, loss=self.losses, loss_weights=self.loss_weights, metrics=["accuracy"])
            
            # Train the model
            model_mt.fit(x_tr, {"softmax_subject": y_tr1, "dense_lr": y_tr2},
                                validation_data=(x_te, {"softmax_subject": y_te1, "dense_lr": y_te2}),
                                epochs=self.epochs, batch_size=self.batch_size)
            history_all.append(model_mt)

            # Define the filepath with ".tf" extension
            filepath = f"eegnet_mt_subj{i}_{timestamp}.tf"
            # Save the model with ".tf" extension
            model_mt.save(os.path.join(self.save_path, filepath))

            # Evaluate the model on the testing data
            evaluation = model_mt.evaluate(x_te, {"softmax_subject": y_te1, "dense_lr": y_te2})
            evaluation_all.append(evaluation)
            
        return history_all, evaluation_all
    
    # this method helps to extract necessary MI classification accuracy from evaluation_all list (because it includes losses, several accuracies)
    def extract_acc_res(self, evaluation_all):
        dense_lr_accuracy_list = [] # motor imagery accuracy
        overall_acc_list = [] # not sure what this represents
        si_acc_list = []
        for evaluation in evaluation_all:
            # Extract the accuracy for the "dense_lr" output
            dense_lr_accuracy = evaluation[4]  # I assume the accuracy for "dense_lr" is at index 4 (will represent MI classification)
            overall_acc = evaluation[0]
            si_acc = evaluation[2] # subject independent accuracy (just if needed)
            dense_lr_accuracy_list.append(dense_lr_accuracy)
            overall_acc_list.append(overall_acc)
            si_acc_list.append(si_acc)
        return dense_lr_accuracy_list


if __name__ == "__main__":
    
    # Define session path and variable saved path
    session_path = '.../Dropbox/DATASETS/OpenBMI_Dataset_Motor_Imagery/downsampled_dataset/session1/'
    save_path_var = '...Dropbox/DATASETS/OpenBMI_Dataset_Motor_Imagery/pickle_dataset/'

    prep_datasets = DataPreprocessor(session_path, save_path_var)

    #to save time please use the saved pickle files for datasets
    # openbci raw dataset, segmented and filtered (not used here)
    #openbci_tr_subj_dep, openbci_te_subj_dep, openbci_label_tr_subj_dep, openbci_label_te_subj_dep = prep_datasets.read_pkl_dataset()

    # LOSO-CV dataset (directly reading previously saved pickled dataset to save time
    #prep_datasets.save_eeg_dataset()
    datasets_subj_indep = prep_datasets.read_eeg_dataset()

    # extracting train/test data from datasets
    x_tr_subj_indep, y_tr_subj_indep, x_te_subj_indep, y_te_subj_indep = [], [], [], []
    for dataset in datasets_subj_indep:
        x_train = dataset.train_ds['x']
        y_train = dataset.train_ds['y']
        x_test = dataset.test_ds['x']
        y_test = dataset.test_ds['y']
        
        x_tr_subj_indep.append(x_train)
        y_tr_subj_indep.append(y_train)
        x_te_subj_indep.append(x_test)
        y_te_subj_indep.append(y_test)

    # init classifiers for single loss classification
    classifiers =  CNNsOneLossClassification('.../results_single_loss/')
    # EEGNet single loss 
    EEGNet_acc_list, EEGNet_history = classifiers.train_model_classification(EEGNet, x_tr_subj_indep, x_te_subj_indep, y_tr_subj_indep, y_te_subj_indep)
    # DeepConvNet single loss
    DeepConvNet_acc_list, DeepConvNet_history = classifiers.train_model_classification(DeepConvNet, x_tr_subj_indep, x_te_subj_indep, y_tr_subj_indep, y_te_subj_indep)
    # ShallowConvNet single loss
    ShallowConvNet_acc_list = classifiers.train_model_classification(ShallowConvNet, x_tr_subj_indep, x_te_subj_indep, y_tr_subj_indep, y_te_subj_indep)


    # init EEGNet Two losses
    eegnet_twoloss = EEGNetTwoLosses('.../results_two_losses/', y_tr_subj_indep, y_te_subj_indep)
    eegnet_twoloss_history, eegnet_twoloss_eval = eegnet_twoloss.model_training(x_tr_subj_indep, x_te_subj_indep)
    eegnet_twoloss_mi_acc = eegnet_twoloss.extract_acc_res(eegnet_twoloss_eval)


    #VITransformer model (will be changed due to slow training)
    '''vit_model = TransformerClassification('.../results_vit_models/')
    vit_model_list = vit_model.init_models(datasets_pkl)
    # transformer classification
    vit_test_res_acc_list, vit_test_res_acc_ave = vit_model.train_and_test_models(vit_model_list)'''
