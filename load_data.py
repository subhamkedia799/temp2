import h5py
import numpy as np
from scipy import fftpack

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
mods = ['8PSK','AM-DSB-WC','BPSK','OOK','GMSK','4ASK','16QAM','64QAM','QPSK','FM']
#'CPFSK' is replaced by 'OOK' and 'PAM4' by '4ASK'
snr_range=[-8,8]
snrs=np.array(range(snr_range[0],snr_range[1]+1,2))
mods=np.array(mods)

file_name = 'dataset/selected_data.hdf5'
Xd = h5py.File(file_name, 'r')

data=Xd['data']
mod_label=Xd['mod_label']
snr_label=Xd['snr_label']

X=[]
Y=[]
lbl=[]
count=0

for ind in range(0, data.shape[0]):
    mod = mods_total[np.argmax(mod_label[ind])]
    snr = snr_label[ind]
    lbl.append((mod,snr))
    Y.append(count)
    count+=1
    
X = np.transpose(data, (0, 2, 1))
del data, mod_label, snr_label

X1=np.transpose(X)
X1=dost(X1)
X1=np.absolute(X1)
X1=np.transpose(X1)
X = (X-np.mean(X,axis=2,keepdims=True))/np.std(X,axis=2,keepdims=True)
X=np.concatenate((X,X1),axis=1)

del X1
print(X.shape)

print('----------------------Dataset Loaded----------------------')

    



