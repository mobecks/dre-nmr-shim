# -*- coding: utf-8 -*-
"""
Created on Tue 15 June 08:26:01 2021

@author: Moritz Becker (IMT)

Script to plot best achievable spectrum shimmed with first-oder only.
"""

import os
import sys
import json
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from scipy import signal

# Import utils script
MYPATH = ''             # TODO: insert path to scripts
DATAPATH = ''           # TODO: insert path for data
#CAUTION: DATAPATH should include ray_results folder !
sys.path.append(MYPATH+'Utils/')
import utils_Spinsolve

import argparse
parser = argparse.ArgumentParser(description="Run 1shot ensemble")
parser.add_argument("--verbose",type=int,default=0)         # 0 for no output, 1 for minimal output, 2 for max output
parser.add_argument("--sample",type=str,default='H2OCu')    #H2OCu, EtOH-0.1, EtOH-0.5
input_args = parser.parse_args()

plt.style.use(['science','high-contrast'])

class Arguments():
    def __init__(self):
        self.count = 0                      # experiment counter
        self.sample = input_args.sample     # sample type

my_arg = Arguments()

    
com = utils_Spinsolve.init( verbose=(input_args.verbose>0) )

# run standard proton experiment without distortions to X,Y,Z (others are zero)
xaxis, initial_spectrum, ref_shims = utils_Spinsolve.setShimsAndRun(com, my_arg.count, [0,0,0], return_shimvalues=True, verbose=(input_args.verbose>1))
linewidth_initial = utils_Spinsolve.get_linewidth_Hz(initial_spectrum)
ref_shims = ref_shims[:3]
my_arg.count += 1

# crop noise in plot
tmp = initial_spectrum
global MIN_IDX
global MAX_IDX 
ns = tmp[int(3*len(tmp)/4):-1]
MIN_IDX, MAX_IDX = np.where(tmp>(ns.mean()+0.5*ns.mean()))[0][0], np.where(tmp>(ns.mean()+0.5*ns.mean()))[0][-1]
w,h,start,stop = signal.peak_widths(tmp, signal.find_peaks(tmp, height = tmp.max()*0.9, distance=1000)[0])
start, stop = (start-2**15/2)/2**15*2e4, (stop-2**15/2)/2**15*2e4
# annotate FWHM in plot
plt.hlines(h,start,stop, colors="grey", linestyles='dotted')
plt.annotate('\scriptsize{}Hz'.format(int(linewidth_initial.item())), [stop+w/2/2**15*2e4,h+50], va='center')
# plot data
plt.plot(xaxis[MIN_IDX:MAX_IDX], initial_spectrum[MIN_IDX:MAX_IDX])
plt.legend()
plt.title('Optimal spectrum without distortions')
plt.ylabel("Signal [a.u.]")
plt.xlabel("Frequency [Hz]")
plt.savefig(DATAPATH + '/DRE/img_dre_{}_best.pdf'.format(my_arg.sample))

utils_Spinsolve.shutdown(com, verbose=(input_args.verbose>0))