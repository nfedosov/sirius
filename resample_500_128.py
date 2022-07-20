# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 01:28:32 2022

@author: Fedosov
"""

import mne
import numpy as np
import pandas as pd
import os


DIR_ROOT = 'C:/Users/Fedosov/PycharmProjects/sirius/results/session_'
#DIR_ROOT2 = 'C:/Users/Fedosov/PycharmProjects/sirius/results/'

source_session = [321,]
target_session = [128,]

runs = [0,1,2]

fs = 500
target_fs = 128



ORIGINAL_CHANNEL_NAMES = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'F3..',  'Fz..',  'F4..', 'P3..', 'Pz..', 'P4..','unk1','unk2']




for subj_ord_idx ,subject_idx in enumerate(source_session):
    for ordered_run_idx, run_idx in enumerate(runs):

        data= pd.read_csv(DIR_ROOT+str(subject_idx)+'/bci_exp_'+str(run_idx)+'/data.csv')
        data = data.to_numpy()
        states = data[:,-1]
        data = data[:,1:-1]
        
        T500 = data.shape[0]
        
        
        info = mne.create_info(29,500)
        
        raw = mne.io.RawArray(data.T, info)
        
        raw = raw.resample(sfreq=128.0)
        fs_real = raw.info['sfreq']
        assert fs_real == 128.
        
        data = raw[:, :][0].T
        
        # print(data.shape)
        df = {ORIGINAL_CHANNEL_NAMES[idx].rstrip('.'): data[:, idx] for idx in range(len(ORIGINAL_CHANNEL_NAMES))}
        ground_truth = np.full(data.shape[0], 0)
        for i in range(T500):
            if round(i*target_fs/fs) != ground_truth.shape[0]:
                idx = round(i*target_fs/fs)
            else:
                idx = round(i*target_fs/fs)-1
            ground_truth[idx] = states[i]
            
        df['state'] = ground_truth
        df = pd.DataFrame(df)
        
        num_exp = ordered_run_idx
        
        results_path = 'results/session_{}/{}_{}/'.format(target_session[subj_ord_idx],'bci_exp',  ordered_run_idx)
        
        
        #import matplotlib.pyplot as plt
        #import scipy.signal as sgn
   
        #signals = df.to_numpy()[:,:-1]
        #states = df.to_numpy()[:,-1]
       
        #rest_idx = states == 1
        #move_idx = states == 3
        
        #f, pxx_rest = sgn.welch(signals[rest_idx,8], fs = 128, nperseg = 128, noverlap = 64, nfft = 128)
        #f, pxx_move = sgn.welch(signals[move_idx,8], fs = 128, nperseg = 128, noverlap = 64, nfft = 128)
        
        
        
        #plt.figure()
        #plt.plot(f,np.log10(pxx_rest))
        #plt.plot(f, np.log10(pxx_move))
        #if ordered_run_idx ==2:
        #    if subject_idx == 2:
        #        plt.figure()
        #        plt.plot(signals)
        
        try:
            os.makedirs(results_path)
        except:
            pass
  

        df.to_csv(results_path+'data.csv')
