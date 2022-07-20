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
from matplotlib import pyplot as plt
import scipy.linalg as la
    
class SimpleNet(nn.Module):
    def __init__(self, in_channels, output_channels, lag_backward, srate):
        super(SimpleNet, self).__init__()
        
        
        ch_names = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'F3..',  'Fz..',  'F4..', 'P3..', 'Pz..', 'P4..']

        self.ch_groups = [list(),list(),list()]
  
        
        for i, ch in enumerate(ch_names):
            ls = [int(s) for s in ch if s.isdigit()]
            if len(ls) == 0:
                self.ch_groups[1].append(i)
                continue
            else:
                dig = int(ls[len(ls)-1])
                if dig %2 == 1:
                    self.ch_groups[2].append(i)
                if dig %2 == 0:
                    self.ch_groups[0].append(i)
                if (dig == 1) | (dig == 2):
                    self.ch_groups[1].append(i)
       
   
        
     
        self.in_channels = in_channels
        
        
        '''
        self.filtering1 = [] # mu, alpha, beta, e. t. c.
        self.batch1 = []
        
        self.filtering2 = []
        #self.filtering2_central = torch.nn.Conv2d(6,6, (, 1), bias = False, groups = 6)
        #abs
        self.batch2 = []
        
        
        self.filtering3 = [] #envelopes of filters
        self.pool1 = [] # scale
        self.batch3 = []
        
        
        
        
        self.features1 = [] 
        # ReLu
        self.pool2 = [] # scale
        self.batch4 = []
        
        self.features2 = []
        # ReLu
        self.pool3 =  []# scale
        self.batch5 = []
        for i in range(3):
            self.filtering1.append(torch.nn.Conv2d(1, 5, (1, 32), bias = False)) # mu, alpha, beta, e. t. c.
            self.batch1.append(torch.nn.BatchNorm2d(5))
            
            self.filtering2.append(torch.nn.Conv2d(5,5, (11, 1), bias = False, groups = 5))
            #self.filtering2_central = torch.nn.Conv2d(6,6, (, 1), bias = False, groups = 6)
            #abs
            self.batch2.append(torch.nn.BatchNorm2d(5))
            
            
            self.filtering3.append(torch.nn.Conv2d(5, 5, (1,32), groups = 5)) #envelopes of filters
            self.pool1.append(torch.nn.AvgPool2d((1,7),stride = (1,3))) # scale
            self.batch3.append(torch.nn.BatchNorm2d(5))
            
            
            
            
            self.features1.append(torch.nn.Conv2d(1,12, (5,8)))   
            # ReLu
            self.pool2.append(torch.nn.MaxPool2d((1,3),stride = (1,2))) # scale
            self.batch4.append(torch.nn.BatchNorm2d(12))
            
            self.features2.append(torch.nn.Conv2d(1,10, (12,8)))
            # ReLu
            self.pool3.append(torch.nn.MaxPool2d((1,3),stride = (1,2)))# scale
            self.batch5.append(torch.nn.BatchNorm2d(10))'''
            
            
            
        ####
        self.filtering1_1 = torch.nn.Conv2d(1, 5, (1, 32), bias = False) # mu, alpha, beta, e. t. c.
        self.batch1_1 = torch.nn.BatchNorm2d(5)
        
        self.filtering2_1 = torch.nn.Conv2d(5,5, (11, 1), bias = False, groups = 5)
        #self.filtering2_central = torch.nn.Conv2d(6,6, (, 1), bias = False, groups = 6)
        #abs
        self.batch2_1 = torch.nn.BatchNorm2d(5)
        
        
        self.filtering3_1 = torch.nn.Conv2d(5, 5, (1,32), groups = 5) #envelopes of filters
        self.pool1_1 = torch.nn.AvgPool2d((1,7),stride = (1,3)) # scale
        self.batch3_1 = torch.nn.BatchNorm2d(5)
        
        
        
        
        self.features1_1 = torch.nn.Conv2d(1,12, (5,8))  
        # ReLu
        self.pool2_1 = torch.nn.MaxPool2d((1,3),stride = (1,2)) # scale
        self.batch4_1 = torch.nn.BatchNorm2d(12)
        
        self.features2_1 = torch.nn.Conv2d(1,10, (12,8))
        # ReLu
        self.pool3_1 = torch.nn.MaxPool2d((1,3),stride = (1,2))# scale
        self.batch5_1 = torch.nn.BatchNorm2d(10)
        
        ####
        ####
        self.filtering1_2 = torch.nn.Conv2d(1, 5, (1, 32), bias = False) # mu, alpha, beta, e. t. c.
        self.batch1_2 = torch.nn.BatchNorm2d(5)
        
        self.filtering2_2 = torch.nn.Conv2d(5,5, (11, 1), bias = False, groups = 5)
        #self.filtering2_central = torch.nn.Conv2d(6,6, (, 1), bias = False, groups = 6)
        #abs
        self.batch2_2 = torch.nn.BatchNorm2d(5)
        
        
        self.filtering3_2 = torch.nn.Conv2d(5, 5, (1,32), groups = 5) #envelopes of filters
        self.pool1_2 = torch.nn.AvgPool2d((1,7),stride = (1,3)) # scale
        self.batch3_2 = torch.nn.BatchNorm2d(5)
        
        
        
        
        self.features1_2 = torch.nn.Conv2d(1,12, (5,8))  
        # ReLu
        self.pool2_2 = torch.nn.MaxPool2d((1,3),stride = (1,2)) # scale
        self.batch4_2 = torch.nn.BatchNorm2d(12)
        
        self.features2_2 = torch.nn.Conv2d(1,10, (12,8))
        # ReLu
        self.pool3_2 = torch.nn.MaxPool2d((1,3),stride = (1,2))# scale
        self.batch5_2 = torch.nn.BatchNorm2d(10)
        
        ####
        
        ####
        self.filtering1_3 = torch.nn.Conv2d(1, 5, (1, 32), bias = False) # mu, alpha, beta, e. t. c.
        self.batch1_3 = torch.nn.BatchNorm2d(5)
        
        self.filtering2_3 = torch.nn.Conv2d(5,5, (11, 1), bias = False, groups = 5)
        #self.filtering2_central = torch.nn.Conv2d(6,6, (, 1), bias = False, groups = 6)
        #abs
        self.batch2_3 = torch.nn.BatchNorm2d(5)
        
        
        self.filtering3_3 = torch.nn.Conv2d(5, 5, (1,32), groups = 5) #envelopes of filters
        self.pool1_3 = torch.nn.AvgPool2d((1,7),stride = (1,3)) # scale
        self.batch3_3 = torch.nn.BatchNorm2d(5)
        
        
        
        
        self.features1_3 = torch.nn.Conv2d(1,12, (5,8))  
        # ReLu
        self.pool2_3 = torch.nn.MaxPool2d((1,3),stride = (1,2)) # scale
        self.batch4_3 = torch.nn.BatchNorm2d(12)
        
        self.features2_3 = torch.nn.Conv2d(1,10, (12,8))
        # ReLu
        self.pool3_3 = torch.nn.MaxPool2d((1,3),stride = (1,2))# scale
        self.batch5_3 = torch.nn.BatchNorm2d(10)
        
        ####
            
            
            
        
      
    
        self.linear1 = torch.nn.Linear(10*3, 10)
        self.batch7 = torch.nn.BatchNorm1d(10)
        #sigmoid
        
    
        self.linear2 = torch.nn.Linear(10, 3)
        #sigmoid
    
    
    
        
    
   
        

    def forward(self, inputs):
        
        y = torch.Tensor()
        
        
            
        x = inputs[:,None,self.ch_groups[0],:]
        x = self.filtering1_1(x)
        x = self.batch1_1(x)
            
        x = self.filtering2_1(x)
        x = torch.abs(x)
        x = self.batch2_1(x)
            
        x = self.filtering3_1(x)
        x = self.pool1_1(x)
        x = self.batch3_1(x)
            
        x = self.features1_1(torch.squeeze(x)[:,None,:,:])
        x = torch.nn.functional.relu(x)
        x = self.pool2_1(x)
        x = self.batch4_1(x)
            
        x = self.features2_1(torch.squeeze(x)[:,None,:,:])
        x = torch.nn.functional.relu(x)
        x = self.pool3_1(x)
        x = self.batch5_1(x)
            
        x = torch.squeeze(torch.mean(x,dim = -1))
            
        y = torch.cat((y,x),dim = 1)
        
        ####
        
        
        x = inputs[:,None,self.ch_groups[1],:]
        x = self.filtering1_2(x)
        x = self.batch1_2(x)
            
        x = self.filtering2_2(x)
        x = torch.abs(x)
        x = self.batch2_2(x)
            
        x = self.filtering3_2(x)
        x = self.pool1_2(x)
        x = self.batch3_2(x)
            
        x = self.features1_2(torch.squeeze(x)[:,None,:,:])
        x = torch.nn.functional.relu(x)
        x = self.pool2_2(x)
        x = self.batch4_2(x)
            
        x = self.features2_2(torch.squeeze(x)[:,None,:,:])
        x = torch.nn.functional.relu(x)
        x = self.pool3_2(x)
        x = self.batch5_2(x)
            
        x = torch.squeeze(torch.mean(x,dim = -1))
            
        y = torch.cat((y,x),dim = 1)
        
        
        ####
        x = inputs[:,None,self.ch_groups[2],:]
        x = self.filtering1_3(x)
        x = self.batch1_3(x)
            
        x = self.filtering2_3(x)
        x = torch.abs(x)
        x = self.batch2_3(x)
            
        x = self.filtering3_3(x)
        x = self.pool1_3(x)
        x = self.batch3_3(x)
            
        x = self.features1_3(torch.squeeze(x)[:,None,:,:])
        x = torch.nn.functional.relu(x)
        x = self.pool2_3(x)
        x = self.batch4_3(x)
            
        x = self.features2_3(torch.squeeze(x)[:,None,:,:])
        x = torch.nn.functional.relu(x)
        x = self.pool3_3(x)
        x = self.batch5_3(x)
            
        x = torch.squeeze(torch.mean(x,dim = -1))
            
        y = torch.cat((y,x),dim = 1)
        
        ####
        
        
        
        x = torch.sigmoid(self.linear1(y))
        x = self.batch7(x)
        
   
        x = torch.sigmoid(self.linear2(x))
        
        
      
        return x
        
        
        
   


class Model:
    def __init__(self):
        self.labels_id = [1,2,3]
        self.srate = 128
        self.b, self.a = sn.butter(2, [5, 36], btype='bandpass', fs=self.srate)
        self.b50, self.a50 = sn.butter(2, [58, 62], btype='bandstop', fs=self.srate)
        #self.b60, self.a60 = sn.butter(2, [58, 62], btype='bandstop', fs=self.srate)
        #self.b60, self.a60 = [np.ones(self.b60.shape), np.ones(self.a60.shape)]
        self.lag_backwards = 128*2
        self.train_batch_size = 1024
        self.val_batch_size = 1024
        self.test_batch_size = 1024
        self.train_sessions = np.arange(1,80,dtype = int)#[1]
        self.train_runs = [0, 1,3,4]
        self.val_sessions = np.arange(1,80,dtype = int)#[1]
        self.val_runs = [2,5]
        self.test_sessions = np.arange(81,110,dtype = int)
        self.test_runs = [0,1,2,3,4,5]
        #self.percentage_of_train = 0.85
        self.epochs = 40
        self.criterion = nn.MSELoss()
        self.device = 'cpu'
        self.train_val_stride = 8# 64#self.lag_backwards // 2
        self.test_stride = 1
        self.train_only_full_class = True
        self.val_only_full_class = True
        self.test_only_full_class = True

    def get_unnormalized_accuracy(self, outputs, labels):
        class_predicted = outputs.cpu().detach().numpy().argmax(axis=1)
        class_labeled = labels.cpu().detach().numpy().argmax(axis=1)
        return np.sum(class_predicted == class_labeled)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(self.train_dataloader):         
            inputs, labels = data
            #indices_to_flip = torch.randint(0, 2, size=(inputs.size(0),), dtype=torch.bool)
            #inputs[indices_to_flip] = torch.flip(inputs[indices_to_flip], dims=(-1, ))

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            self.model.eval()
            running_accuracy += self.get_unnormalized_accuracy(outputs, labels)

        return running_loss / (i+1), running_accuracy / len(self.train_dataloader.dataset)

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
                                         b=self.b, a50=self.a50, b50=self.b50,device=self.device)
        data_storage_val = DataStorage(session_nums=self.val_sessions, run_nums=self.val_runs, a=self.a,
                                       b=self.b, a50=self.a50, b50=self.b50, device=self.device)
        # data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage,
        #                     stride=self.train_val_stride, valid_indices=None, shuffle=True)
        # train_data, val_data = data.split_ds(percentage_of_first=self.percentage_of_train)
        train_data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage_train,
                                  stride=self.train_val_stride, valid_indices=None, shuffle=True,
                                  only_full_class=self.train_only_full_class)
        val_data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage_val,
                                 stride=self.train_val_stride, valid_indices=None, shuffle=True,
                                 only_full_class=self.val_only_full_class)
        self.train_dataloader = DataLoader(train_data, batch_size=self.train_batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val_data, batch_size=self.val_batch_size, shuffle=False)

        self.model = SimpleNet(data_storage_train.data.shape[1], len(self.labels_id), self.lag_backwards, self.srate)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)#weight_decay=1e-2)#0.02#3e-4

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
        #plt.figure()
        #plt.plot(train_accuracy_history)
        #plt.plot(val_accuracy_history)
        

        torch.save(self.model.state_dict(), 'model_weights.pt')

    def test(self):

        data_storage = DataStorage(session_nums=self.test_sessions, run_nums=self.test_runs, a=self.a,
                                   b=self.b, a50=self.a50, b50=self.b50, device=self.device)
        data = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.lag_backwards, data_storage=data_storage,
                            stride=self.test_stride, valid_indices=None, shuffle=True,
                            only_full_class=self.test_only_full_class)
        self.test_dataloader = DataLoader(data, batch_size=self.test_batch_size, shuffle=False)
        self.model = SimpleNet(data_storage.data.shape[1], len(self.labels_id), self.lag_backwards, self.srate)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load('model_weights.pt'))

        print("Trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        print("Total params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        test_running_accuracy = self.get_accuracy_score(self.test_dataloader)

        print(f'Test accuracy: {test_running_accuracy}')


class DataStorage:
    def __init__(self, session_nums, run_nums, a, b, a50, b50, device):
        data = list()
        labels = list()

        cum_num =  0
        for session_num in session_nums:
            for num in run_nums:
                dataframe = pd.read_csv(f'results/session_{session_num}/bci_exp_{num}/data.csv')
                data.append(dataframe.to_numpy()[:, 1:-1])
                data[cum_num] = sn.lfilter(b, a, data[cum_num], axis=0)
                data[cum_num] = sn.lfilter(b50, a50, data[cum_num], axis=0)
                
                cov = np.cov(data[cum_num].T)
                #data[cum_num] = data[cum_num]@la.fractional_matrix_power(cov,-1/2)
                data[cum_num] /= np.sqrt(np.mean(data[cum_num]**2, axis = 0))[None,:]
                labels.append((dataframe.to_numpy()[:, -1]).astype('int'))
                
                cum_num += 1

        self.data = torch.tensor(np.concatenate(data), dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.concatenate(labels), device=device)


class TorchDataset(Dataset):
    def __init__(self, labels_id, amount_of_lags, data_storage, stride, valid_indices=None, shuffle=False, only_full_class=False):
        super(TorchDataset, self).__init__()
        self.labels_id = labels_id
        self.data_storage = data_storage
        self.stride = stride
        self.only_full_class = only_full_class
        if valid_indices is None:
            #boundaries = np.zeros(self.data_storage.labels, dtype=np.int32)
            #boundaries[1:] = self.data_storage.labels[1:] - self.data_storage.labels[:-1]
            #classes_indices = np.isin(data_storage.labels, self.labels_id).astype(int)
            self.valid_indices = []
            for idx in range(amount_of_lags - 1, data_storage.labels.size(0)):
                if data_storage.labels[idx] in self.labels_id:
                    if self.only_full_class:
                        labels_slice = data_storage.labels[idx - amount_of_lags + 1:idx + 1]
                        diff = torch.abs(labels_slice[1:] - labels_slice[:-1])
                        diff = torch.sum(diff).item()
                        if diff == 0:
                            self.valid_indices.append(idx)
                    else:
                        self.valid_indices.append(idx)
        else:
            self.valid_indices = valid_indices
        #boundaries =
        
        # transition = classes_indices[1:]-classes_indices[:-1]
        
        # strided_indices = []
        
        # i = 0
        
        # while i < transition.shape[0]:
        #     if transition[i]==1:
        #         if np.sum(transition[i:i+amount_of_lags] == -1) == 0:
        #             i += amount_of_lags
        #             strided_indices.append(i)
        #         while(np.sum(transition[i:i+self.stride] == -1) == 0):
        #             i += self.stride
        #             strided_indices.append(i)
                
        #     else:
        #         i += 1
            
        
        #0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2,
        
        #0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0
        
        
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
                           stride=self.stride, valid_indices=indices1, shuffle=self.shuffled, 
                           only_full_class=self.only_full_class)
        ds2 = TorchDataset(labels_id=self.labels_id, amount_of_lags=self.amount_of_lags, data_storage=self.data_storage,
                           stride=self.stride, valid_indices=indices2, shuffle=self.shuffled,
                           only_full_class=self.only_full_class)
        return ds1, ds2



model = Model()
model.fit()
model.test()


