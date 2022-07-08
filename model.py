# загружаем библиотеки, необходимые для работы нейросети
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.autograd import Variable
import torch, torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg

import scipy
import sklearn

import numpy as np

import random

from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
import pandas as pd

# %matplotlib inline
import mne
import numpy as np
from mne.preprocessing import (ICA, read_ica)
import scipy.linalg as la
import scipy.signal as sn
from mne.decoding import UnsupervisedSpatialFilter

from scipy.io import loadmat

from matplotlib import pyplot as plt
import math

import os




class envelope_detector(nn.Module):
    def __init__(self, in_channels, channels_per_channel):
        super(self.__class__, self).__init__()
        self.FILTERING_SIZE = 32
        self.ENVELOPE_SIZE = 16
        self.CHANNELS_PER_CHANNEL = channels_per_channel
        self.OUTPUT_CHANNELS = self.CHANNELS_PER_CHANNEL * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.OUTPUT_CHANNELS, bias=False, kernel_size=self.FILTERING_SIZE,
                                        groups=in_channels)
        self.conv_envelope = nn.Conv1d(self.OUTPUT_CHANNELS, self.OUTPUT_CHANNELS, kernel_size=self.ENVELOPE_SIZE,
                                       groups=self.OUTPUT_CHANNELS)
        self.conv_envelope.requires_grad = False
        self.pre_envelope_batchnorm = torch.nn.BatchNorm1d(self.OUTPUT_CHANNELS, affine=False)
        self.conv_envelope.weight.data = (
                    torch.ones(self.OUTPUT_CHANNELS * self.ENVELOPE_SIZE) / self.FILTERING_SIZE).reshape(
            (self.OUTPUT_CHANNELS, 1, self.ENVELOPE_SIZE))
        self.relu = torch.nn.ReLU()
        self.intermidiate = None

    def forward(self, x):
        x = self.conv_filtering(x)
        x = self.pre_envelope_batchnorm(x)
        x = torch.abs(x)
        x = self.conv_envelope(x)
        return x


# создание нейросети
class simple_net(nn.Module):
    def __init__(self, in_channels, output_channels, lag_backward):
        super(self.__class__, self).__init__()
        
        self.in_channels = in_channels
        self.ICA_CHANNELS = 4
        self.fin_layer_decim = 8
        self.CHANNELS_PER_CHANNEL = 1

        self.total_input_channels = self.ICA_CHANNELS  # + in_channels
        self.lag_backward = lag_backward
   
      
        self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)

        self.detector = envelope_detector(self.total_input_channels, self.CHANNELS_PER_CHANNEL)
        
        self.final_out_features = self.ICA_CHANNELS*((lag_backward-self.detector.FILTERING_SIZE-\
                                self.detector.ENVELOPE_SIZE+2)//self.fin_layer_decim)

        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.unmixed_batchnorm = torch.nn.BatchNorm1d(self.total_input_channels, affine=False)

        self.wights_second = nn.Linear(self.final_out_features, output_channels)

        self.pre_out = None
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        all_inputs = self.ica(inputs)

        all_inputs = self.unmixed_batchnorm(all_inputs)

        detected_envelopes = self.detector(all_inputs)

        features = detected_envelopes[:, :, (self.lag_backward-self.detector.FILTERING_SIZE-\
                                self.detector.ENVELOPE_SIZE+2)%self.fin_layer_decim::self.fin_layer_decim].contiguous()
        features = features.view(features.size(0), -1)
        features = self.features_batchnorm(features)
        self.pre_out = features.cpu().data.numpy()
        output = self.sigmoid(self.wights_second(features))
        return output




class Model:
    def __init__(self):
        self.labels_id = [1,2,3]
        self.SRATE = 128 #500
           
        self.b, self.a = sn.butter(2, [2, 40], btype = 'bandpass', fs = self.SRATE)
        self.b50, self.a50 = sn.butter(2, [48,50], btype = 'bandstop', fs = self.SRATE)
        self.b60, self.a60 = sn.butter(2, [58,62], btype = 'bandstop', fs = self.SRATE)
        self.LAG_BACKWARD = 256 #250 
      

    def fit(self): 
        
        #session_num_list = np.setdiff1d(np.arange(1,50,dtype = 'int'), [3,5,10,7,8,4,2,30])
        session_num_list = [1,]
        train_num_list = [0,1,3,4]#np.arange(0,6,dtype = 'int')
       
        #self.model= Model()
        self.data = list()
        train_labels = list()
        
        for session_num in session_num_list:
            
            self.path_dir = 'results/session_{}/'.format(session_num)
            for ordernum, num in enumerate(train_num_list):
                dataframe = pd.read_csv(self.path_dir+'bci_exp_'+str(num)+'/data.csv')
                self.data.append(dataframe.to_numpy()[:,1:-1])
                self.data[ordernum] = sn.lfilter(self.b,self.a, self.data[ordernum], axis = 0)
                self.data[ordernum] = sn.lfilter(self.b50,self.a50, self.data[ordernum], axis = 0)
                self.data[ordernum] = sn.lfilter(self.b60,self.a60, self.data[ordernum], axis = 0)
                self.data[ordernum] = self.data[ordernum]/np.sqrt(np.mean(self.data[ordernum]**2))
                train_labels.append((dataframe.to_numpy()[:,-1]).astype('int'))
        
        
            
        
        self.data = np.concatenate(self.data)#[:100000]
        train_labels = np.concatenate(train_labels)#[:100000]
        
        val_indices = random.sample(np.arange(self.data.shape[0]).tolist(), self.data.shape[0]//4)
        train_indices = [ix for ix in range(self.data.shape[0]) if ix not in val_indices]
        
        self.data_val = self.data[val_indices]
        self.data = self.data[train_indices]
        val_labels = train_labels[val_indices]
        train_labels = train_labels[train_indices]
                
        
        
        batch_size = 64
        
      
        
     
        self.y = np.tile(train_labels[:, np.newaxis], (1, len(self.labels_id)))
        for i in range(self.y.shape[1]):
            self.y[:, i] = (self.y[:, i] == self.labels_id[i]).astype(int)
        self.val_y = np.tile(val_labels[:, np.newaxis], (1, len(self.labels_id)))
        for i in range(self.val_y.shape[1]):
            self.val_y[:, i] = (self.val_y[:, i] == self.labels_id[i]).astype(int)
      
     
        
        # нормализация данных
        # stdX = np.std(X, axis=0)
        # X = X / stdX
        
        #NON_REST_BOOL_VECTOR = y > 0
        #active2full = np.arange(X.shape[0])
        #active2full = active2full[NON_REST_BOOL_VECTOR]
        
        #X_active = X[NON_REST_BOOL_VECTOR]
        #Y_active = np.zeros((np.sum(NON_REST_BOOL_VECTOR), len(event_touse)))
        #for i in range(1, Y_active.shape[1] + 1):
        #    Y_active[y[NON_REST_BOOL_VECTOR] == i, i - 1] = 1
        
            
    
    


        self.net = simple_net(self.data.shape[1], self.y.shape[1], self.LAG_BACKWARD)
        #for i in range(model.ICA_CHANNELS):
            #model.ica.weight.data[0, :, 0] = torch.Tensor([int(i in CHANNELS) for i in range(64)])
    
    
    
        print("Trainable params: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        print("Total params: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        loss_function = nn.MSELoss()#nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0002)#3e-4)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
        
        
        loss_history = []
        max_test_corr = 0
        
        
        #pbar = tqdm()
        
        while True:
            x_batch, y_batch, batch_idx = self.data_generator2(batch_size, 'train')
            #### Train
            self.net.train()
            
            assert x_batch.shape[0] == y_batch.shape[0]
            x_batch = Variable(torch.FloatTensor(x_batch))
            y_batch = Variable(torch.FloatTensor(y_batch))
            optimizer.zero_grad()
        
            y_predicted = self.net(x_batch)
            assert y_predicted.shape[0] == y_batch.shape[0]
        
            loss = loss_function(y_predicted, y_batch)
            loss.backward()
            optimizer.step()
            #y_batch = y_batch.softmax(dim=1)
            loss_history.append(np.mean(y_predicted.cpu().detach().numpy().argmax(axis=1) == y_batch.cpu().detach().numpy().argmax(axis=1)))    
            #pbar.update(1)
            eval_lag = min(100, len(loss_history))
            #pbar.set_postfix(loss = np.mean(loss_history[-eval_lag:]), val_loss=max_test_corr)
        
          
        
            if (len(loss_history)-1) % 20 == 0:     
                         
                plt.figure()
                plt.plot(loss_history)
                x_batch, _, batch_idx  = self.data_generator2(batch_size, 'val')
                    
                self.net.eval()
        
                x_batch = Variable(torch.FloatTensor(x_batch))
                y_predicted = self.net(x_batch).cpu().data.numpy()
                plt.figure()
                plt.plot(y_predicted)
                assert x_batch.shape[0]==y_predicted.shape[0]
               
                val_acc = np.mean(np.squeeze(y_predicted).argmax(axis=1) == self.val_y[batch_idx,:].argmax(axis = 1))
                train_acc = loss_history[-1]
        
                print("Correlation  train {} val {}".format(train_acc, val_acc))
        
            if len(loss_history)>2000:#40000
                break
            
        #Сохранение модели
        torch.save(self.net, 'model.pth')
       
        
        
        
        
        
    def test(self): 
        
        session_num_list = [1,]#[3,5,10,7,8,4,2,30]
        test_num_list = [2,5]#np.arange(0,6,dtype = 'int')
        
        batch_test_size = 5000
        
        self.model= Model()
        self.data = list()
        test_labels = list()
        
        for session_num in session_num_list:
            self.path_dir = 'results/session_{}/'.format(session_num)
            for ordernum, num in enumerate(test_num_list):
                dataframe = pd.read_csv(self.path_dir+'bci_exp_'+str(num)+'/data.csv')
                self.data.append(dataframe.to_numpy()[:,1:-1])
                self.data[ordernum] = sn.lfilter(self.b,self.a, self.data[ordernum], axis = 0)
                self.data[ordernum] = sn.lfilter(self.b50,self.a50, self.data[ordernum], axis = 0)
                self.data[ordernum] = sn.lfilter(self.b60,self.a60, self.data[ordernum], axis = 0)
                self.data[ordernum] = self.data[ordernum]/np.sqrt(np.mean(self.data[ordernum]**2))
                test_labels.append((dataframe.to_numpy()[:,-1]).astype('int'))
                
                
            
        
        self.data = np.concatenate(self.data)
        test_labels = np.concatenate(test_labels)
        
# Обучение нейросети
        
     
        self.y = np.tile(test_labels[:, np.newaxis], (1, len(self.labels_id)))
        for i in range(self.y.shape[1]):
            self.y[:, i] = (self.y[:, i] == self.labels_id[i]).astype(int)
      


        self.net = torch.load('model.pth')
        
       
                
        x_batch, _, batch_idx  = self.data_generator2(batch_test_size)
                    
        self.net.eval()
        
        x_batch = Variable(torch.FloatTensor(x_batch))
        y_predicted = self.net(x_batch).softmax(axis=1).cpu().data.numpy()
        assert x_batch.shape[0]==y_predicted.shape[0]
  
            
        max_test_corr = np.mean(np.squeeze(y_predicted).argmax(axis=1) == self.y[batch_idx,:].argmax(axis = 1))
        
        print("Correlation val on test", max_test_corr)
        
        
        
    def load_model(self):
        self.net = torch.load('model.pth')   # it is 
        self.mem_data = np.zeros((self.net.LAG_BACKWARD, self.net.in_channels))
        self.zi = np.zeros(len(self.a)+len(self.b) - 2)
        self.zi50
        self.net.eval()
        
      
        
        
    
    def predict_once(self, new_samples):
    
        # возможно, ещё понадобится 150 Гц фильтр
        filtered50, self.zi50 = sn.lfilter(self.b50, self.a50, new_samples, axis=0, zi=self.zi50)
        filtered60, self.zi60 = sn.lfilter(self.b, self.a, filtered50, axis=0, zi=self.zi60)
        filtered, self.zi = sn.lfilter(self.b, self.a, filtered60, axis=0, zi=self.zi)
        
        self.mem_data = np.roll(shift = -filtered.shape[0], axis = 0)
        self.mem_data[:filtered.shape[0]] = filtered
        
     
        x = Variable(torch.FloatTensor((self.mem_data.T[np.newaxis,:,:])))
        y = self.net(x).cpu().data.numpy()
        
        self.labels_id
        return  self.labels_id[np.argmax(y)]
        
        
    
            
            
    # функция, нарезающая данные на батчи для обучения
    def data_generator2(self, batch_size, data_tag):
      
        #total_lag = self.LAG_BACKWARD #+1
        #all_batches = math.ceil((X_active.shape[0]) / batch_size)
        if data_tag == 'train':
            y = self.y
            data = self.data
        elif data_tag == 'val':
            y = self.val_y
            data = self.data_val
        available_idx = np.where(np.sum(y, axis = 1)>0)[0][::64]
       
      
        batch_idx = np.array(random.sample(available_idx.tolist(),batch_size))
        batch_cut_idx = batch_idx[:,np.newaxis]-\
                       np.arange(self.LAG_BACKWARD, dtype = int)[np.newaxis,:]
   
   
        batch_x = data[batch_cut_idx,:].transpose(0,2,1)
        batch_y = y[batch_idx,:].astype('float32')
        
        
        return (batch_x, batch_y, batch_idx)
     
        
        
#model = Model()
#model.fit()
#model.test()
#model.run_online()