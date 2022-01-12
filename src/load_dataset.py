import numpy as np
from scipy.sparse import data
from tables.atom import EnumAtom
import ninapro_utils 
from pathlib import Path 
from zipfile import ZipFile
import matplotlib.pyplot as plt 
import collections
import deepdish as dd 
from tqdm import tqdm
from utils import *

import os
import yaml
import random 
import sys

# Read the configuration file for importing model configurations
config_path = Path(__file__).parents[1] / 'config.yml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)

db2_path = Path(__file__).parents[1] / 'data/raw/DB2_zip/'
db2_extract_dir = Path(__file__).parents[1] / 'data/raw/DB2/'
db2_savepath = Path(__file__).parents[1] / 'data/interm/DB2.h5'

os.makedirs(db2_extract_dir, exist_ok=True)
os.makedirs(Path(__file__).parents[1] / 'data/interm/', exist_ok=True)
        
###############################################################
# Extract invidual zip files of subjects second dataset of DB2
###############################################################
with skip_run('skip', 'Extract the second dataset from Ninapro DB2') as check, check():
    for file in tqdm(os.listdir(db2_path)):
        # extract if the file is a zip file
        if (file.split(".")[1] == 'zip') and (file.split("_")[0] == 'DB2'):
            filepath = os.path.join(db2_path, file)
            with ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(db2_extract_dir)

##################################################################################################
# Run the following chunk of code if the exercise 1 (17 classes) should be saved into hdf5 file 
##################################################################################################
with skip_run('skip', 'Import DB2 dataset and save it into HDF5 format') as check, check():
    data_dict = collections.defaultdict()
    for subject in tqdm(np.arange(1, config['num_subjects']+1)):
        # Get EMG, repetition and movement data, cap max length of rest data before and after each movement to 5 seconds
        # Capping occurs by reducing the size of repetition segments since splitting is based on repetition number
        sub_dict = ninapro_utils.import_db2(db2_extract_dir, subject, rest_length_cap=5)

        data_dict['S'+str(subject)] = sub_dict
        
    dd.io.save(db2_savepath, data_dict)

##################################################################
# Run the following code to split the raw dataset and normalize it 
##################################################################
batch_data_path = Path(__file__).parents[1] / 'data/interm/DB2_batch.h5'
with skip_run('skip', 'Create EMG samples of constant length windows (wind_size x channels)') as check, check():   
    window_len = int(np.round(config['window_size'] * config['fs']))
    n_channels = config['n_channels']
    overlap_perct = config['window_overlap']
    
    data = dd.io.load(db2_savepath)
    batch_data = collections.defaultdict()

    for subject in tqdm(np.arange(1, config['num_subjects']+1)):
        rep_regions = data['S'+str(subject)]['rep_regions']
        emg = data['S'+str(subject)]['emg']
        
        # apply mu-law transformation and then normalize the data
        emg = ninapro_utils.mu_law_transformation(emg)
        emg = ninapro_utils.minmax_normalization(emg)
        
        emg_splits = []
        for i in range(0, len(rep_regions), 2):
            rep_len = rep_regions[i+1] - rep_regions[i]
            assert window_len < rep_len, "The repetition data is too short"
            
            num_segs = int((np.floor(rep_len / window_len) - 1) / (1 - overlap_perct))
            step_size = int((1 - overlap_perct) * window_len)
            last_ind = int(rep_regions[i] + window_len * ( 1 + (1 - overlap_perct) *num_segs))

            emg_temp = emg[rep_regions[i]:last_ind, :]
            
            # split each repetition of emg into small windows of constant window_len
            for j in range(0, emg_temp.shape[0] - window_len + 1, step_size):
                emg_splits.append(np.expand_dims(emg_temp[j:j+window_len, :], axis=0))
            
        emg_splits = np.concatenate(emg_splits, axis=0)
            
        batch_data['S'+str(subject)] = emg_splits
        
    dd.io.save(batch_data_path, batch_data)
        
##################################################################
# Split train and test data
##################################################################
with skip_run('skip', 'Creating batches of dataset using random subjects and samples for training the encoder') as check, check():
    data = dd.io.load(batch_data_path)
        
    # shuffled subject list 
    subjects = [15, 7, 32, 9, 8, 18, 37, 31, 3, 35, 19, 36, 33, 6, 11, 29, 1, 34, 21, 17, 24, 25, 27, 13,
                23, 12, 20, 16, 30, 10, 5, 28, 2, 26, 40, 4, 22, 38, 39, 14]
    
    # shuffle the samples of each subject 
    for sub in subjects:
        data['S'+str(sub)] = ninapro_utils.shuffle_emg_samples(data['S'+str(sub)])
        
    n = config['batch_param']['n'] # number of subjects required in each batch
    m = config['batch_param']['m'] # number of samples per subject in each batch

    train_feats, train_labels = [], []
    test_feats, test_labels = [], []
    valid_feats, valid_labels = [], [] # samples removed from the training set (These can be used for validation)

    train_subjects = subjects[:-2]
    test_subjects = subjects[-2:]

    while len(train_subjects) > n:
        batch_subjects = random.sample(train_subjects, k=n)
        batch_samples, batch_labels = [], []
        for subject in batch_subjects:
            emg = data['S'+str(subject)]
            
            batch_samples.append(emg[:m, :, :])
            batch_labels.append(subject * np.ones(m, dtype=np.int8))
                        
            # delete the already retrieved samples from the subject dictionary
            emg = np.delete(emg, np.arange(m), axis=0)
            data['S'+str(subject)] = emg            
        
        batch_samples = np.concatenate(batch_samples, axis=0)
        batch_labels = np.concatenate(batch_labels, axis=0)
        
        train_feats.append(batch_samples)
        train_labels.append(batch_labels)
        
        delete_subjects, valid_feats, valid_labels = ninapro_utils.prune_subject_list(train_subjects, data, n, m, valid_feats, valid_labels)
        
        # update the subject list
        train_subjects = list(set(train_subjects) - set(delete_subjects))
        
        # subjects to be removed due to shortage of samples
        for sub in delete_subjects:
            del data['S'+str(sub)]
    
    train_feats   = np.concatenate(train_feats, axis=0)
    train_labels  = np.concatenate(train_labels, axis=0)
    
    for sub in data.keys():
        valid_feats.append(data[sub])
        
        labels = int(sub.split('S')[1]) * np.ones(data[sub].shape[0], dtype=np.int8)
        valid_labels.append(labels)
    
    valid_feats   = np.concatenate(valid_feats, axis=0)
    valid_labels  = np.concatenate(valid_labels, axis=0)
    
    # prepare the test dataset 
    for subject in subjects[-2:]:
        emg = data['S'+str(subject)]
        labels = subject * np.ones(emg.shape[0], dtype=np.int8)

        test_feats.append(emg)
        test_labels.append(labels)

    test_feats = np.concatenate(test_feats, axis=0)
    test_labels  = np.concatenate(test_labels, axis=0)
    
    traindata_savepath = Path(__file__).parents[1] / 'data/processed/Train.h5'
    testdata_savepath = Path(__file__).parents[1] / 'data/processed/Test.h5'
    validdata_savepath = Path(__file__).parents[1] / 'data/processed/Validation.h5'
    
    # save the training data in terms of the created batches
    # train_data = collections.defaultdict()
    # for i, val in enumerate(train_feats):
    #     train_data['batch_'+str(i)] = {'features': train_feats[i],
    #                                     'labels': train_labels[i]}
        
    dd.io.save(traindata_savepath, {'features': train_feats,
                                    'labels': train_labels})
    
    dd.io.save(testdata_savepath, {'features': test_feats,
                                    'labels': test_labels})
        
    dd.io.save(validdata_savepath, {'features': valid_feats,
                                    'labels': valid_labels})
            


plt.show()

