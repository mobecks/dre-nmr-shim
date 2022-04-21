# Author: https://github.com/mobecks
import os
import pickle

# install dev version: python -m pip install git+https://github.com/jjhelmus/nmrglue.git@6ca36de7af1a2cf109f40bf5afe9c1ce73c9dcdc

import sys
MYPATH = ''                         #TODO: insert path to scripts
DATASET_PATH = ''                   #TODO: insert path to ShimDB
sys.path.append(MYPATH+'Utils/')
import utils_IO

DATAPATH = MYPATH+'/data/'
if not os.path.exists(DATAPATH):
    os.mkdir(DATAPATH)

DOWNSAMPLEFACTOR = 16 # data size reduction

#%% Creating batched input
data_all, labels_all, xaxis = utils_IO.get_dataset(DATASET_PATH, target_def = 'firstorder',
                                                   normalize=False, downsamplefactor=DOWNSAMPLEFACTOR)

data_tmp, labels_tmp, dic = utils_IO.batch_dataset(data_all, labels_all, downsamplefactor=DOWNSAMPLEFACTOR
                                                   , channels=4, offsets=[1000], sets = [21**3])

# dump numpy array to pickle
with open(os.path.dirname(os.path.abspath(DATASET_PATH))+'/preloaded_pickle/batched_data_2048p.pickle','wb') as f:
    dic = {'offsets':1000, 'channels': 21**3, 'sets': 'poc'}
    pickle.dump([data_tmp, labels_tmp, dic], f)

# load whole package from disc. Save iteration time.
with open(os.path.dirname(os.path.abspath(DATASET_PATH))+'/preloaded_pickle/batched_data_2048p.pickle','rb') as f:
    [data_tmp, labels_tmp, dic] = pickle.load(f)
