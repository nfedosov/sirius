# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:51:42 2022

@author: Fedosov
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.signal as sn
from torch.nn.utils import parametrize
import pandas as pd
    
    
class Symmetric(nn.Module):
    def forward(self, weight):
        left_kernel_size = weight.size(-1) // 2
        right_kernel_size = left_kernel_size
        if weight.size(-1) % 2 == 1:
            left_kernel_size += 1        
        weight_left = weight[:, :, :left_kernel_size]
        weight_right = torch.flip(weight[:, :, :right_kernel_size], dims=(-1,))
        full_weight = torch.cat([weight_left, weight_right], dim=-1)
        return full_weight
        

class EnvelopeDetector(nn.Module):
    def __init__(self, in_channels, channels_per_channel):
        super(EnvelopeDetector, self).__init__()
        self.FILTERING_SIZE = 64
        self.ENVELOPE_SIZE = 64
        
        self.DROPOUT_P = 0#0.25
        
        self.CHANNELS_PER_CHANNEL = channels_per_channel
        self.OUTPUT_CHANNELS = self.CHANNELS_PER_CHANNEL * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.OUTPUT_CHANNELS, bias=False, kernel_size=self.FILTERING_SIZE,
                                        groups=in_channels)
        parametrize.register_parametrization(self.conv_filtering, "weight", Symmetric())
        
        self.conv_envelope = nn.Conv1d(self.OUTPUT_CHANNELS, self.OUTPUT_CHANNELS, kernel_size=self.ENVELOPE_SIZE,
                                       groups=self.OUTPUT_CHANNELS)
        self.conv_envelope.requires_grad = False
        self.pre_envelope_batchnorm = torch.nn.BatchNorm1d(self.OUTPUT_CHANNELS, affine=False)
        self.conv_envelope.weight.data = (
                torch.ones(self.OUTPUT_CHANNELS * self.ENVELOPE_SIZE) / self.FILTERING_SIZE).reshape(
            (self.OUTPUT_CHANNELS, 1, self.ENVELOPE_SIZE))
                    
        self.dropout = nn.Dropout(p=self.DROPOUT_P)

    def forward(self, x):
        x = self.conv_filtering(x)
        
        x = self.dropout(x)
        
        x = self.pre_envelope_batchnorm(x)
        x = torch.abs(x)
        x = self.conv_envelope(x)
        
        x = self.dropout(x)
        
        return x


class SimpleNet(nn.Module):
    def __init__(self, in_channels, output_channels, lag_backward):
        super(SimpleNet, self).__init__()
        self.ICA_CHANNELS = 4
        self.fin_layer_decim = 20
        self.CHANNELS_PER_CHANNEL = 1
        
        self.DROPOUT_P = 0#0.25

        self.total_input_channels = self.ICA_CHANNELS  # + in_channels
        self.lag_backward = lag_backward

        self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)

        self.detector = EnvelopeDetector(self.total_input_channels, self.CHANNELS_PER_CHANNEL)

        self.final_out_features = self.ICA_CHANNELS * ((lag_backward - self.detector.FILTERING_SIZE - \
                                                        self.detector.ENVELOPE_SIZE + 2) // self.fin_layer_decim)

        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.unmixed_batchnorm = torch.nn.BatchNorm1d(self.total_input_channels, affine=False)

        self.wights_second = nn.Linear(self.final_out_features, output_channels)
        
        self.dropout = nn.Dropout(p=self.DROPOUT_P)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        all_inputs = self.ica(inputs)
        
        all_inputs = self.dropout(all_inputs)

        all_inputs = self.unmixed_batchnorm(all_inputs)

        detected_envelopes = self.detector(all_inputs)

        features = detected_envelopes[:, :, (self.lag_backward - self.detector.FILTERING_SIZE - \
                                             self.detector.ENVELOPE_SIZE + 2) % self.fin_layer_decim::self.fin_layer_decim].contiguous()
        features = features.view(features.size(0), -1)
        features = self.features_batchnorm(features)
        
        features = self.dropout(features)
        
        output = self.wights_second(features)
        # print(output)
        #output = self.sigmoid(output)
        # print(output)
        return output
    

# class SimpleNet(nn.Module):
#     def __init__(self, in_channels, output_channels, lag_backward):
#         super(SimpleNet, self).__init__()
#         self.flatten = nn.Flatten()
#         kernel_size = 33
#         self.conv1d_space = nn.Conv1d(in_channels, 3, kernel_size=1)
#         self.activation = nn.Tanh()
#         self.conv1d = nn.Conv1d(3, 3, kernel_size=kernel_size, groups=3)
#         # self.pool1 = nn.AvgPool1d(kernel_size=65, stride=32)
#         size_after_convo = lag_backward - kernel_size + 1
#         # self.linear = nn.Linear(in_channels * size_after_convo, output_channels)
        
#     def forward(self, inputs):
#         out = self.conv1d_space(inputs)
#         out = self.conv1d(out)
#         out = torch.mean(out ** 2, axis=-1)
#         out = self.flatten(out)
#         return out


class Model:
    def __init__(self):
        self.labels_id = [1,2,3]
        self.srate = 128
        self.b, self.a = sn.butter(2, [2, 40], btype='bandpass', fs=self.srate)
        self.b50, self.a50 = sn.butter(2, [48, 52], btype='bandstop', fs=self.srate)
        self.b60, self.a60 = sn.butter(2, [58, 62], btype='bandstop', fs=self.srate)
        self.lag_backwards = 256
        self.train_batch_size = 64
        self.val_batch_size = 64
        self.test_batch_size = 512
        self.train_sessions = np.arange(1,80,dtype = int)#[1]
        self.train_runs = [0, 1,3,4]
        self.val_sessions = [1, 2]
        self.val_runs = [2,5]
        self.test_sessions = np.arange(81,110,dtype = int)
        self.test_runs = [0,1,2,3,4,5]
        #self.percentage_of_train = 0.85
        self.epochs = 50
        self.criterion = nn.CrossEntropyLoss()
        self.device = 'cpu'
        self.train_val_stride = 32#self.lag_backwards // 2
        self.test_stride = 1

    def get_unnormalized_accuracy(self, outputs, labels):
        class_predicted = outputs.softmax(axis=1).cpu().detach().numpy().argmax(axis=1)
        class_labeled = labels.cpu().detach().numpy().argmax(axis=1)
        return np.sum(class_predicted == class_labeled)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(self.train_dataloader):         
            inputs, labels = data
            indices_to_flip = torch.randint(0, 2, size=(inputs.size(0),), dtype=torch.bool)
            inputs[indices_to_flip] = torch.flip(inputs[indices_to_flip], dims=(-1, ))

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            self.model.eval()
            running_accuracy += self.get_unnormalized_accuracy(outputs, labels)

        return running_loss / i, running_accuracy / len(self.train_dataloader.dataset)

    def get_accuracy_score(self, dataloader):
        self.model.eval()
        running_accuracy = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = self.model(inputs)
            running_accuracy += self.get_unnormalized_accuracy(outputs, labels)

        return running_accuracy / len(dataloader.dataset)

    def fit(self):
        # data_storage = DataStorage(session_nums=self.train_val_sessions, run_nums=self.train_val_runs, a=self.a,
        #                            b=self.b,
        #                            a50=self.a50, b50=self.b50, a60=self.a60, b60 = self.b60, device=self.device)
        data_storage_train = DataStorage(session_nums=self.train_sessions, run_nums=self.train_runs, a=self.a,
                                         b=self.b, a50=self.a50, b50=self.b50, a60=self.a60, b60 = self.b60, device=self.device)
        data_storage_val = DataStorage(session_nums=self.val_sessions, run_nums=self.val_runs, a=self.a,
                                       b=self.b, a50=self.a50, b50=self.b50, a60=self.a60, b60 = self.b60, device=self.device)
        # data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage,
        #                     stride=self.train_val_stride, valid_indices=None, shuffle=True)
        # train_data, val_data = data.split_ds(percentage_of_first=self.percentage_of_train)
        train_data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage_train,
                                  stride=self.train_val_stride, valid_indices=None, shuffle=True)
        val_data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage_val,
                                 stride=self.train_val_stride, valid_indices=None, shuffle=True)
        self.train_dataloader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_data, batch_size=self.val_batch_size, shuffle=False)

        self.model = SimpleNet(data_storage_train.data.shape[1], len(self.labels_id), self.lag_backwards)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4 )#weight_decay=1e-2)#0.02

        print("Trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("Total params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        train_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        for epoch in range(self.epochs):

            running_loss, running_accuracy = self.train_one_epoch()

            train_loss_history.append(running_loss)
            train_accuracy_history.append(train_accuracy_history)

            val_running_accuracy = self.get_accuracy_score(self.val_dataloader)
            val_accuracy_history.append(val_running_accuracy)

            print(f'Epoch num: {epoch + 1} Train error: {running_loss}, train accuracy: {running_accuracy} '
                  f'Validation accuracy: {val_running_accuracy}')

        print('Finished Training')

        torch.save(self.model.state_dict(), 'model_weights.pt')

    def test(self):

        data_storage = DataStorage(session_nums=self.test_sessions, run_nums=self.test_runs, a=self.a,
                                   b=self.b, a50=self.a50, b50=self.b50, a60=self.a60, b60 = self.b60, device=self.device)
        data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage,
                            stride=self.test_stride, valid_indices=None, shuffle=True)
        self.test_dataloader = DataLoader(data, batch_size=self.test_batch_size, shuffle=False)
        self.model = SimpleNet(data_storage.data.shape[1], len(self.labels_id), self.lag_backwards)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('model_weights.pt'))

        print("Trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("Total params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        test_running_accuracy = self.get_accuracy_score(self.test_dataloader)

        print(f'Test accuracy: {test_running_accuracy}')


class DataStorage:
    def __init__(self, session_nums, run_nums, a, b, a50, b50, a60, b60, device):
        data = list()
        labels = list()

        for session_num in session_nums:
            for ordernum, num in enumerate(run_nums):
                dataframe = pd.read_csv(f'results/session_{session_num}/bci_exp_{num}/data.csv')
                data.append(dataframe.to_numpy()[:, 1:-1])
                data[ordernum] = sn.lfilter(b, a, data[ordernum], axis=0)
                data[ordernum] = sn.lfilter(b50, a50, data[ordernum], axis=0)
                data[ordernum] = sn.lfilter(b60, a60, data[ordernum], axis=0)
                expectation = np.mean(data[ordernum], axis=0)[None, :]
                standard_d = np.std(data[ordernum], axis=0)[None, :]
                data[ordernum] = data[ordernum] - expectation
                data[ordernum] = data[ordernum] / standard_d
                labels.append((dataframe.to_numpy()[:, -1]).astype('int'))

        self.data = torch.tensor(np.concatenate(data), dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.concatenate(labels), device=device)


class TorchDataset(Dataset):
    def __init__(self, labels_id, amount_of_lags, data_storage, stride, valid_indices=None, shuffle=False):
        super(TorchDataset, self).__init__()
        self.labels_id = labels_id
        self.data_storage = data_storage
        self.stride = stride
        if valid_indices is None:
            self.valid_indices = np.where(np.isin(data_storage.labels, self.labels_id))[0]
            self.valid_indices = self.valid_indices[self.valid_indices >= (amount_of_lags - 1)]
        else:
            self.valid_indices = valid_indices
        self.valid_indices = np.sort(self.valid_indices)
        strided_indices = [self.valid_indices[0]]
        for idx in self.valid_indices[1:]:
            if (idx - strided_indices[-1]) >= self.stride:
                strided_indices.append(idx)
        self.valid_indices = torch.tensor(strided_indices)
        if shuffle:
            self.valid_indices = self.valid_indices[torch.randperm(self.valid_indices.size(0))]
        self.labels_id = torch.tensor(self.labels_id)
        self.shuffled = shuffle
        self.len = self.valid_indices.shape[0]
        self.amount_of_lags = amount_of_lags

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        data_slice = self.data_storage.data[valid_idx-self.amount_of_lags+1:valid_idx+1].T
        answer_idx = self.data_storage.labels[valid_idx] - 1
        y = torch.zeros(3, dtype=torch.float32)
        y[answer_idx] = 1.
        return data_slice, y

    def split_ds(self, percentage_of_first):
        amount_of_data1 = int(np.ceil(percentage_of_first * self.len))
        indices1 = np.random.choice(self.valid_indices, size=amount_of_data1, replace=False)
        indices2 = np.setdiff1d(self.valid_indices.numpy(), indices1)
        if not self.shuffled:
            indices1 = np.sort(indices1)
            indices2 = np.sort(indices2)
        ds1 = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.amount_of_lags, data_storage=self.data_storage,
                           stride=self.stride, valid_indices=indices1, shuffle=self.shuffled)
        ds2 = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.amount_of_lags, data_storage=self.data_storage,
                           stride=self.stride, valid_indices=indices2, shuffle=self.shuffled)
        return ds1, ds2



model = Model()
model.fit()
model.test()


