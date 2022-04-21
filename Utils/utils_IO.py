
################################################################
# Use nmrglue package
# latest version (09.06.2021) needs this commit https://github.com/jjhelmus/nmrglue.git@6ca36de7af1a2cf109f40bf5afe9c1ce73c9dcdc
################################################################

import numpy as np
import nmrglue as ng
import os
import glob

#Function to read ShimDB from folder structure into numpy array
def get_dataset(DATASET_PATH, target_def = 'firstorder', normalize=True, downsamplefactor=1):

    norm_factor = 2**15
    #load data, sort and remove first point
    FNAMEs_shims = glob.glob(DATASET_PATH+"/*/shims.par")
    FNAMEs_shims = sorted(FNAMEs_shims, key = os.path.getmtime)[1:] #sort by creation time
    FNAMEs_1d = glob.glob(DATASET_PATH+"/*/data.1d")
    FNAMEs_1d = sorted(FNAMEs_1d, key = os.path.getmtime)[1:] #sort by creation time
    
    # extract shim values to numpy array
    shims_dic = np.array([ { line.split()[0] : int(line.split()[2]) for line in open(f) }
                 for f in FNAMEs_shims])

    DIR_ref_shims = glob.glob(DATASET_PATH+"/RefShims.par")
    if DIR_ref_shims == []:
        print('No ref sim file found. Old version of dataset acquisition. Exiting.' )
        exit()
    else:
        ref_shim = np.array([ { line.split()[0] : int(line.split()[2]) for line in open(DIR_ref_shims[0]) }])[0]
    
    #dummy scan to get parameters
    dummy_dic,_  = ng.spinsolve.read(os.path.dirname(FNAMEs_1d[0]))

    data_bin = np.empty([len(FNAMEs_1d), int(dummy_dic['spectrum']['xDim']/downsamplefactor) ])
    for idx,f in enumerate(FNAMEs_1d):
        dic, fid = ng.spinsolve.read(os.path.dirname(f))
        # more uniform listing
        udic = ng.spinsolve.guess_udic(dic, fid)
        # fft and phase correction
        spectrum = ng.proc_base.fft(fid)
        # phase correct with given values
        spectrum = ng.proc_base.ps(spectrum, p0=float(dic['proc']['p0Phase']) ,p1=float(dic['proc']['p1Phase']))
        spectrum = ng.proc_base.di(spectrum)
        #downsample
        data_bin[idx] = np.array(spectrum)[::downsamplefactor]
        
    xaxis = np.arange(-udic[0]['sw']/2,udic[0]['sw']/2, udic[0]['sw']/udic[0]['size'])

    if target_def == 'firstorder': # return 3 first order shim values per sample
        #create array of labels similar to digital twin (as offset to reference)
        labels = np.empty([len(shims_dic),  3])
        for idx,d in enumerate(shims_dic):
            # change absolute to relative shim values wrt reference shim
            labels[idx,0] = d["xshim"] - ref_shim["xshim"]
            labels[idx,1] = d["yshim"] - ref_shim["yshim"]
            labels[idx,2] = d["zshim"] - ref_shim["zshim"]          
        if normalize:
            # normalize data and labels
            labels = labels / norm_factor
            max_data = 1e5
            data_all = data_bin/max_data          
            return data_all, labels, norm_factor, max_data, xaxis
        else:
            return data_bin, labels, xaxis
            
# function to combine 4 spectra (channels) to a single input batch
def batch_dataset(data_all, labels_all, downsamplefactor, channels=4, offsets=[1000], sets = [21**3]):
    data_tmp = []
    labels_tmp = []

    # get data and labels as array
    for s in sets:
        data_tmp.append(np.zeros([s, channels, int(32768/downsamplefactor)]))
        labels_tmp.append(np.zeros([s, 3]))

    counters = [0,0,0]
    for idx_off, off in enumerate(offsets):
        start = sum(sets[:idx_off])
        stop = start + sets[idx_off]
        # for each unique label -> search DB for spectra with offset in X,Y,Z
        for idx_sub, l in enumerate(labels_all[start:stop]):
            idx = idx_sub+start
            bool_x = np.sum((labels_all == (l + [off,0,0])), axis=1)    # does the shim values equal the current shim setting + offset in X ?
            idx_x = np.where(bool_x == 3)[0]                            # get index

            bool_y = np.sum((labels_all == (l + [0,off,0])), axis=1)
            idx_y = np.where(bool_y == 3)[0]

            bool_z = np.sum((labels_all == (l + [0,0,off])), axis=1)
            idx_z = np.where(bool_z == 3)[0]
            if idx_x.any() != False and idx_y.any() != False and idx_z.any() != False : # check whether all 3 offsets are found, else discard this data instance
                if len(idx_x) > 1:
                    idx_x = min(idx_x, key=lambda x:abs(x-idx)) # get closest idx if more than 2 spectra found
                if len(idx_y) > 1:
                    idx_y = min(idx_y, key=lambda x:abs(x-idx)) # get closest idx if more than 2 spectra found
                if len(idx_z) > 1:
                    idx_z = min(idx_z, key=lambda x:abs(x-idx)) # get closest idx if more than 2 spectra found
                    # extract the four spectra for unshimmed u, u+X, u+Y, u+Z and put into structured array
                data_tmp[idx_off][counters[idx_off]] = [data_all[idx], data_all[idx_x.item()], data_all[idx_y.item()], data_all[idx_z.item()]]
                labels_tmp[idx_off][counters[idx_off]] = l
                counters[idx_off] += 1

    # trim zeros values because some data instance do not have all 3 offsets and are discarded
    for idx, s in enumerate(sets):
        #last value
        lastnonzero = counters[idx]

        labels_tmp[idx] = labels_tmp[idx][:lastnonzero]
        data_tmp[idx] = data_tmp[idx][:lastnonzero]
    
    # info description
    dic = {'offsets':offsets, 'channels': channels, 'sets': sets}
    
    return data_tmp, labels_tmp, dic
    