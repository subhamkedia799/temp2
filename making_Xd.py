import numpy as np
import h5py

mods = ['32PSK','16APSK','32QAM','FM','GMSK','32APSK','OQPSK','8ASK','BPSK','8PSK','AM-SSB-SC','4ASK','16PSK','64APSK','128QAM','128APSK','AM-DSB-SC','AM-SSB-WC','64QAM','QPSK','256QAM','AM-DSB-WC','OOK','16QAM']
mods_filt=['8PSK','AM-DSB-WC','BPSK','OOK','GMSK','4ASK','16QAM','64QAM','QPSK','FM']
#'CPFSK' is replaced by 'OOK' and 'PAM4' by '4ASK'
snr_range=[-20,20]
snrs=np.array(range(snr_range[0],snr_range[1]+1,2))

file_name = 'dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
DATASET = h5py.File(file_name, 'r')

DATASET['X']=np.array(DATASET['X'])

Xd={}

for mod in mods:
  for snr in snrs:
    Xd[(mod,snr)]=[]

for ind in range(0,Xd['X'].shape[0]):
  mod_index=np.argmax(np.array(Xd['Y'][ind]))
  snr=Xd['Z'][ind]
  Xd[(mods[mod_index],snr)].append(list(np.transpose(DATASET['X'][ind])))
  
del DATASET
print(list(Xd.keys()))
