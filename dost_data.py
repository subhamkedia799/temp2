import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TENSORFLOW_FLAGS"]  = "device=gpu"
import numpy as np
import tensorflow as th
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import keras.models as models
from keras.layers import Reshape,Dense,Dropout,Activation,Flatten,GaussianNoise,Conv2D,MaxPooling2D,AveragePooling2D
from keras.regularizers import *
from keras.optimizers import adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle as cPickle
import random, sys, keras
from scipy import fftpack
import datetime
time=str(datetime.datetime.now())
import h5py


def dost_bw(l):
    out = np.zeros(int(2*np.log2(l)))
    l1 = np.arange(np.log2(l)-2,-1,-1)
    l2 = np.arange(0,np.log2(l)-1)
    out[1:int(1+np.log2(l)-1)]=l1
    out[-int(np.log2(l)-1):]=l2
    out = np.exp2(out).astype(np.int16)
    return out

def dost(inp):
    l = inp.shape[0]
    fft_inp = fftpack.fftshift(fftpack.fft(fftpack.ifftshift(inp,axes=0),axis=0),axes=0)
    #plt.figure(figsize = (30,5))
    #ax = np.linspace(-512,511,2**10)
    #plt.plot(ax,fft_inp[0,:])
    bw_inp = dost_bw(l)
    #print(bw_inp)
    k = 0
    dost_inp = np.zeros_like(fft_inp)
    for r in bw_inp:
        if(r==1):
            dost_inp[k,...] = fft_inp[k,...]
            k = k+r
        else:
            dost_inp[k:r+k,...] = fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(fft_inp[k:r+k,...],axes=0),axis=0),axes=0)
            k = k+r
    #plt.plot(fft_inp)
    #plt.figure(figsize = (20,5))
    #plt.plot(np.abs(dost_inp[0,:]))
    return dost_inp

mods_total = ['32PSK','16APSK','32QAM','FM','GMSK','32APSK','OQPSK','8ASK','BPSK','8PSK','AM-SSB-SC','4ASK','16PSK','64APSK','128QAM','128APSK','AM-DSB-SC','AM-SSB-WC','64QAM','QPSK','256QAM','AM-DSB-WC','OOK','16QAM']
mods = ['8PSK','AM-DSB-WC','BPSK','16APSK','64APSK','GMSK','16QAM','64QAM','QPSK','FM']

snr_range=[-8,8]
snrs=list(range(snr_range[0],snr_range[1]+1,2))

mod_map={}
for i in range(0,len(mods)):
    itemindex = mods_total.index(mods[i])
    mod_map[itemindex] = i



file_name = 'dataset/RML2018_selected_data.hdf5'
Xd = h5py.File(file_name, 'r')

data=Xd['data']
mod_label=Xd['mod_label']
snr_label=Xd['snr_label']

X=[]
Y=[]
lbl=[]


for ind in range(0, data.shape[0]):
    mod_index = np.argmax(mod_label[ind])
    mod = mods_total[mod_index]
    snr = snr_label[ind]
    snr_index = snrs.index(snr)
    lbl.append((mod,snr))
    Y.append(mod_map[mod_index]*len(snrs) + snr_index)
    
    
X = np.transpose(data, (0, 2, 1))
del data, mod_label, snr_label

X1=np.empty(X.shape)
X0=X[:,0,:]+X[:,1,:]*1j
X0=np.transpose(X0)
X0=dost(X0)
X0=np.transpose(X0)
X1[:,0,:]=np.real(X0)
X1[:,1,:]=np.imag(X0)
data_file = h5py.File('dataset/RML2018_selected_data_dost.hdf5', 'w')
data_file.create_dataset('dost_data', data=X1)
data_file.close()
