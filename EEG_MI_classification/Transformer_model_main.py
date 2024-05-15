# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:16:46 2024

@author: madina.kudaibergenova

Transformer model code.

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import lightning as L
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import time
import gc
# Generate a timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")

# version for subject dependent dataset
class EEGDataset(data.Dataset):
     def __init__(self, x_tr, x_te, y_tr, y_te):
         super().__init__()
         self.train_ds = {
             'x': x_tr,
             'y': y_tr,
         }
         self.test_ds = {
             'x': x_te,
             'y': y_te,
         }

     def __len__(self):
         return len(self.dataset['x'])

     def __getitem__(self, idx):
         x = self.dataset['x'][idx]
         y = self.dataset['y'][idx]
         x = torch.tensor(x).float()
         y = torch.tensor(y).unsqueeze(-1).float()
         return x, y

     def split(self, __split):
         self.__split = __split
         return self

     @property
     def dataset(self):
         assert self.__split is not None, "Please specify the split of dataset!"
         if self.__split == "train":
             return self.train_ds
         elif self.__split == "test":
             return self.test_ds
         else:
             raise TypeError("Unknown type of split!")      
'''
# version for 6 subject OpenBCI subject independent
class EEGDataset_indep(data.Dataset):
    def __init__(self, x_tr, x_te, y_tr, y_te, target_subject):
        super().__init__()
        self.target_subject = target_subject
        self.train_ds = {
            'x': self._concat_data_except_target(x_tr, x_te, target_subject),
            'y': self._concat_label_except_target(y_tr, y_te, target_subject),
        }
        self.test_ds = {
            'x': torch.tensor(x_te[target_subject]).float(),  # Convert to tensor
            'y': torch.tensor(y_te[target_subject][0,:]).float(),  # Convert to tensor
        }
        self.__split = None

    def _concat_data_except_target(self, data_tr, data_te, target_subject):
        # Concatenate train and test data of subjects excluding the target subject
        concatenated_data = []
        for i, tr_data in enumerate(data_tr):
            if i != target_subject:
                concatenated_data.append(torch.tensor(tr_data).float())  # Convert to tensor
        for i, te_data in enumerate(data_te):
            if i != target_subject:
                concatenated_data.append(torch.tensor(te_data).float())  # Convert to tensor
        # Concatenate and reshape the data
        concatenated_data = torch.cat(concatenated_data, dim=0)
        concatenated_data = torch.transpose(concatenated_data, 1, 2)
        # Run garbage collection
        gc.collect()
        return concatenated_data
    
    def _concat_label_except_target(self, label_tr, label_te, target_subject):
        # Concatenate train and test data of subjects excluding the target subject
        concatenated_label = []
        for i, tr_label in enumerate(label_tr):
            if i != target_subject:
                concatenated_label.append(torch.tensor(tr_label[0,:]).float())  # Convert to tensor
        for i, te_label in enumerate(label_te):
            if i != target_subject:
                concatenated_label.append(torch.tensor(te_label[0,:]).float())  # Convert to tensor
        # Concatenate and reshape the data
        concatenated_label = torch.cat(concatenated_label, dim=0)
        concatenated_label = concatenated_label.view(-1)  # Reshape to (num_samples)
        # Run garbage collection
        gc.collect()
        return concatenated_label

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, idx):
        x = self.dataset['x'][idx]
        y = self.dataset['y'][idx]
        x = torch.tensor(x).float()
        y = torch.tensor(y).unsqueeze(-1).float()
        # Run garbage collection
        gc.collect()
        return x, y

    def split(self, __split):
        self.__split = __split
        return self

    @property
    def dataset(self):
        assert self.__split is not None, "Please specify the split of dataset!"
        if self.__split == "train":
            return self.train_ds
        elif self.__split == "test":
            return self.test_ds
        else:
            raise TypeError("Unknown type of split!")
'''
# LOSO-CV fr 54 subjects, independent
class EEGDataset_LOSO_CV:
    def __init__(self, x_tr, x_te, y_tr, y_te, target_subject):
        self.target_subject = target_subject
        self.train_ds = {
            'x': self._concat_data_except_target(x_tr, x_te, target_subject),
            'y': self._concat_label_except_target(y_tr, y_te, target_subject),
        }
        self.test_ds = {
            'x': x_te[target_subject].astype(np.float32),  # Convert to float32
            'y': y_te[target_subject][0, :].astype(np.float32),  # Extract a single array and convert to float32
        }
        self.__split = None

    def _concat_data_except_target(self, data_tr, data_te, target_subject):
        # Concatenate train and test data of subjects excluding the target subject
        concatenated_data = []
        for i, tr_data in enumerate(data_tr):
            if i != target_subject:
                concatenated_data.append(tr_data.astype(np.float32))  # Convert to float32
        for i, te_data in enumerate(data_te):
            if i != target_subject:
                concatenated_data.append(te_data.astype(np.float32))  # Convert to float32
        # Concatenate and transpose the data
        concatenated_data = np.concatenate(concatenated_data, axis=0)
        concatenated_data = np.transpose(concatenated_data, (0, 1, 2))
        # Run garbage collection if needed
        gc.collect()
        return concatenated_data

    def _concat_label_except_target(self, label_tr, label_te, target_subject):
        # Concatenate train and test labels of subjects excluding the target subject
        concatenated_label = []
        for i, tr_label in enumerate(label_tr):
            if i != target_subject:
                concatenated_label.append(tr_label[0, :].astype(np.float32))  # Convert to float32
        for i, te_label in enumerate(label_te):
            if i != target_subject:
                concatenated_label.append(te_label[0, :].astype(np.float32))  # Convert to float32
        # Concatenate and reshape the labels
        concatenated_label = np.concatenate(concatenated_label, axis=0)
        concatenated_label = concatenated_label.reshape(-1)  # Reshape to (num_samples,)
        # Run garbage collection if needed
        gc.collect()
        return concatenated_label
    
    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, idx):
        x = self.dataset['x'][idx]
        y = self.dataset['y'][idx]
        x = torch.tensor(x).float()
        y = torch.tensor(y).unsqueeze(-1).float()
        # Run garbage collection
        gc.collect()
        return x, y

    def split(self, __split):
        self.__split = __split
        return self

    @property
    def dataset(self):
        assert self.__split is not None, "Please specify the split of dataset!"
        if self.__split == "train":
            return self.train_ds
        elif self.__split == "test":
            return self.test_ds
        else:
            raise TypeError("Unknown type of split!")

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.losses = []

    def update(self, val):
        self.losses.append(val)

    def show(self):
        out = torch.mean(
            torch.stack(
                self.losses[np.maximum(len(self.losses)-self.num, 0):]
            )
        )
        return out  
   
  
#wrapping model for subj dependent and independent classification
class ModelWrapper(L.LightningModule):
    def __init__(self, arch, dataset, batch_size, lr, max_epoch):
        super().__init__()

        self.arch = arch
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch

        self.train_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.automatic_optimization = False

        self.train_loss_recorder = AvgMeter()
        self.train_acc_recorder = AvgMeter()

    def forward(self, x):
        return self.arch(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute().data.cpu()

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.train_loss_recorder.update(loss.data)
        self.train_acc_recorder.update(acc)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

        self.log("train_loss", self.train_loss_recorder.show(), prog_bar=True)
        self.log("train_acc", self.train_acc_recorder.show(), prog_bar=True)
        self.train_loss_recorder.reset()
        self.train_acc_recorder.reset()

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.test_accuracy.update(y_hat, y)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True, logger=True)
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.split("train"),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.split("test"),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(self.max_epoch * 0.25),
                    int(self.max_epoch * 0.5),
                    int(self.max_epoch * 0.75),
                ],
                gamma=0.1
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]


###################### EEG CLassification Model
class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x #+ self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x
    
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=19)

class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
        return x
    
def initialize_trainer(SEED, CHECKPOINT_DIR, MAX_EPOCH):
    """Initialize trainer with logger, callbacks, and settings."""
    tensorboardlogger = TensorBoardLogger(save_dir=CHECKPOINT_DIR + f"{timestamp}" + "logs/")
    csvlogger = CSVLogger(save_dir=CHECKPOINT_DIR + f"{timestamp}" + "logs/")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(
        monitor='train_acc',
        dirpath=CHECKPOINT_DIR,
        mode='max',
    )
    early_stopping = EarlyStopping(
        monitor="train_acc", min_delta=0.00, patience=3, verbose=False, mode="max"
    )

    seed_everything(SEED, workers=True)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCH,
        logger=[tensorboardlogger, csvlogger],
        callbacks=[lr_monitor, checkpoint, early_stopping],
        log_every_n_steps=1,
    )
    return trainer