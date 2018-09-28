import h5py
import numpy as np
from scipy import fftpack
from numpy import zeros, newaxis

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
 
 
mods = ['32PSK','16APSK','32QAM','FM','GMSK','32APSK','OQPSK','8ASK','BPSK','8PSK','AM-SSB-SC','4ASK','16PSK','64APSK','128QAM','128APSK','AM-DSB-SC','AM-SSB-WC','64QAM','QPSK','256QAM','AM-DSB-WC','OOK','16QAM']
mods_filt=['8PSK','AM-DSB-WC','BPSK','OOK','GMSK','4ASK','16QAM','64QAM','QPSK','FM']
#'CPFSK' is replaced by 'OOK' and 'PAM4' by '4ASK'
snr_range=[-8,8]
snrs=np.array(range(snr_range[0],snr_range[1]+1,2))
mods=np.array(mods)
mods_filt=np.array(mods_filt)

file_name = 'dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
Xd = h5py.File(file_name, 'r')

X = []  
Y = []
#lbl= np.zeros((Xd['X'].shape[0],(len(snrs)*len(mods_filt))),dtype=int)
lbl=[]
count=0

for ind in range(0,len(Xd['X'])):
    mod=mods[np.argmax(np.array(Xd['Y'][ind]))]
    #snr_index=np.where(snrs == Xd['Z'][ind])
    #mod_snr_index=len(snrs)*mod_index+snr_index
    #lbl[ind][mod_snr_index]=1
    snr=Xd['Z'][ind]
    
    if mod in mods_filt and snr in snrs:
        X2=np.array(Xd['X'][ind])
        X2=X2[:, :, newaxis]
        X1=dost(X2)
        X1=np.absolute(X1)
        X1=np.transpose(X1)
        X2=np.transpose(X2)
        #X2 = (X2-np.mean(X2,axis=2,keepdims=True))/np.std(X2,axis=2,keepdims=True)
        X.append(np.concatenate((X2,X1),axis=1))
        lbl.append((mod,snr))
        Y.append(count)
        count+=1

X = np.vstack(X)

del Xd
print(X.shape)
