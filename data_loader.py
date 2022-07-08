import mne
import numpy as np
import pandas as pd
import os


DIR_ROOT = 'C:/Users/Fedosov/Downloads/eeg-motor-movementimagery-dataset-1.0.0/files/'
#DIR_ROOT2 = 'C:/Users/Fedosov/PycharmProjects/sirius/results/'

ORIGINAL_CHANNEL_NAMES = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'F3..',  'Fz..',  'F4..', 'P3..', 'Pz..', 'P4..']
GT_MAP_FISTS = {'T0': 0, 'T1': 1, 'T2': 3}
GT_MAP_FEET = {'T0': 0, 'T1': 0, 'T2': 2}



FISTS_DATASETS = [ 4, 8, 12]
FEET_DATASETS = [6, 10, 14]

for subject_idx in range(1, 110):
    for ordered_run_idx, run_idx in enumerate(FISTS_DATASETS+FEET_DATASETS):

        folder_head = 'S{0:03d}'.format(subject_idx)
        experiment_num = 'R{0:02d}'.format(run_idx)
        dir_path = f'{DIR_ROOT}{folder_head}/{folder_head}{experiment_num}.edf'
        if run_idx in FISTS_DATASETS:
            gt_map = GT_MAP_FISTS
        elif run_idx in FEET_DATASETS:
            gt_map = GT_MAP_FEET
        else:
            raise ValueError
        print(dir_path)
        raw = mne.io.read_raw_edf(dir_path)
        fs = raw.info['sfreq']
        if fs == 128:
            print('128 detected')
        if fs != 128:
            raw = raw.resample(sfreq=128.0)
        fs = raw.info['sfreq']
        assert fs == 128.
        data = raw[ORIGINAL_CHANNEL_NAMES, :][0].T
        assert len(ORIGINAL_CHANNEL_NAMES) == data.shape[1]
        # print(data.shape)
        df = {ORIGINAL_CHANNEL_NAMES[idx].rstrip('.'): data[:, idx] for idx in range(len(ORIGINAL_CHANNEL_NAMES))}
        ground_truth = np.full(data.shape[0], -1)
        for annotation in raw.annotations:
            start = annotation['onset']
            finn = start + annotation['duration']
            start = int(round(start * fs))
            finn = int(round(finn * fs))
            gt_point = annotation['description']
            ground_truth[start:finn] = gt_map[gt_point]
        df['state'] = ground_truth
        df = pd.DataFrame(df)
        
        num_exp = ordered_run_idx
        
        results_path = 'results/session_{}/{}_{}/'.format(subject_idx,'bci_exp',  num_exp)
        
        
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
