import numpy as np
import matplotlib.pyplot as plt

# returns the desired standard deviation to construct a normal distribution 
# for a predefined SNR, given an input array
# noise = np.random.normal(loc=0, scale= get_noise_avg(d.detach().numpy(), snr = 1) , size = d.shape)
def get_noise_avg(pred_numpy, snr = 1):
    power = pred_numpy **2
    avg = np.mean(power)
    db = 10 * np.log10(avg)
    noise_avg_db = db - snr
    noise_avg = 10 **(noise_avg_db / 10)
    return np.sqrt(noise_avg)
  
    
# %% Scores and Statistics #################################################################################################

# Return the SNR of a spectrum (as in Magritek)
# Signal is the maximum peak.
# Noise is the rms of the last 1/4 of the spectrum.  
def getSNR(array):   
    ry = array.real
    sig = max(ry)
    ns = ry[int(3*len(array)/4):-1]
    return sig/np.std(ns)
