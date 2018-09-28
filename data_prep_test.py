import h5py
import numpy as np
 
mods = ['32PSK','16APSK','32QAM','FM','GMSK','32APSK','OQPSK','8ASK','BPSK','8PSK','AM-SSB-SC','4ASK','16PSK','64APSK','128QAM','128APSK','AM-DSB-SC','AM-SSB-WC','64QAM','QPSK','256QAM','AM-DSB-WC','OOK','16QAM']
mods_filt=['8PSK','AM-DSB-WC','BPSK','OOK','GMSK','4ASK','16QAM','64QAM','QPSK','FM']
#'CPFSK' is replaced by 'OOK' and 'PAM4' by '4ASK'
snr_range=[-8,8]
snrs=np.array(range(snr_range[0],snr_range[1]+1,2))
mods=np.array(mods)
mods_filt=np.array(mods_filt)

file_name = 'dataset/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
Xd = h5py.File(file_name, 'r')
new_Xd={'X':[],'Y':[],'Z':[]}

for ind in range(0,len(Xd['X'])):
    mod=mods[np.argmax(np.array(Xd['Y'][ind]))]
    snr=Xd['Z'][ind]
    
    if mod in mods_filt and snr in snrs:
        new_Xd['X'].append(Xd['X'][ind])
        new_Xd['Y'].append(Xd['Y'][ind])
        new_Xd['Z'].append(Xd['Z'][ind])

del Xd

data_file = h5py.File('dataset/selected_data.hdf5', 'w')
data_file.create_dataset('my_dataset', data=new_Xd)
data_file.close()

del new_Xd
