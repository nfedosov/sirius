# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 23:49:03 2022

@author: Fedosov
"""

import pandas as pd
import numpy as np
import scipy.signal as sn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

session_num = 3
num_list = [0,1,2,3,4,5]
srate = 128


b, a = sn.butter(2, [9, 13], btype='bandpass', fs=srate)
b50, a50 = sn.butter(2, [48, 52], btype='bandstop', fs=srate)
b60, a60 = sn.butter(2, [58, 62], btype='bandstop', fs=srate)

data = list()
states = list()
for num in num_list:
    dataframe = pd.read_csv(f'results/session_{session_num}/bci_exp_{num}/data.csv')
    data.append(dataframe.to_numpy()[:, 1:-1])
    states.append(dataframe.to_numpy()[:, -1])
    data[num] = sn.lfilter(b, a, data[num], axis=0)
    data[num] = sn.lfilter(b50, a50, data[num], axis=0)
    data[num] = sn.lfilter(b60, a60, data[num], axis=0)
    
data = np.concatenate(data)
states = np.concatenate(states)

state1_idx = states == 1
state2_idx = states == 2
state3_idx = states == 3

available_idx  = state1_idx | state2_idx | state3_idx

data1 = data[state1_idx,:]
data2 = data[state2_idx,:]
data3 = data[state3_idx,:]

cov1 = np.cov(data1.T)
cov2 = np.cov(data2.T)
cov3 = np.cov(data3.T)
common_cov = cov1+cov2+cov3


import scipy.linalg as la
l1,w1= la.eigh(cov1, common_cov)
l2,w2= la.eigh(cov2, common_cov)
l3,w3= la.eigh(cov3, common_cov)


comp1 = data@w1[:,0]
comp2 = data@w2[:,0]
comp3 = data@w3[:,0]

feat_red = np.zeros((np.sum(available_idx),3))
win = 256
for i in range(feat_red.shape[0]):
    feat_red[i,0] = np.log(np.sum(comp1[(np.where(available_idx)[0])[i]-win:(np.where(available_idx)[0])[i]]**2))
for i in range(feat_red.shape[0]):
    feat_red[i,1] = np.log(np.sum(comp2[(np.where(available_idx)[0])[i]-win:(np.where(available_idx)[0])[i]]**2))
for i in range(feat_red.shape[0]):
    feat_red[i,2] = np.log(np.sum(comp3[(np.where(available_idx)[0])[i]-win:(np.where(available_idx)[0])[i]]**2))
    
    
  
    
states_red = states[available_idx]

import random
train_idx = random.sample(np.arange(np.shape(feat_red)[0], dtype = int).tolist(), round(np.shape(feat_red)[0]*7/8))
test_idx = np.setdiff1d(np.arange(np.shape(feat_red)[0], dtype = int),np.array(train_idx))
#train_idx = np.arange(np.shape(feat_red)[0], dtype = int)[:round(np.shape(feat_red)[0]*7/8)]
#test_idx = np.arange(np.shape(feat_red)[0], dtype = int)[round(np.shape(feat_red)[0]*7/8):]



lda = LDA()
lda.fit(feat_red[train_idx], states_red[train_idx])

states_predicted =lda.predict(feat_red[test_idx])

acc = np.mean(states_predicted ==states_red[test_idx])

print(acc)


















