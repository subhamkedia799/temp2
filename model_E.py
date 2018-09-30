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

X1=np.transpose(X)
X1=dost(X1)
X1=np.absolute(X1)
X1=np.transpose(X1)
X=np.concatenate((X,X1),axis=1)

del X1
print(X.shape)


print('----------------------Dataset Loaded----------------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42,stratify=Y)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=42,stratify=Y_train)

Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)
Y_val=to_categorical(Y_val)
del X,Y

print('----------------------Dataset Splitted----------------------')
in_shp = list(X_train.shape[1:])
print("Dataset dimension ",X_train.shape, in_shp)
classes = Y_train.shape[1]

# Build VT-CNN2 Neural Net model using Keras primitives -- 
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape((in_shp+[1]), input_shape=in_shp))
model.add(Conv2D(128, (4, 5), activation='relu', name='conv1', padding='same', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(AveragePooling2D(pool_size=(1, 4), strides=None, padding='valid', data_format=None))
model.add(Conv2D(128, (4, 5), activation='relu', name='conv2', padding='same', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(AveragePooling2D(pool_size=(1, 4), strides=None, padding='valid', data_format=None))
model.add(Conv2D(128, (4, 5), activation='relu', name='conv3', padding='same', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None))
model.add(Conv2D(128, (4, 5), activation='relu', name='conv4', padding='same', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None))
model.add(Conv2D(128, (4, 7), activation='relu', name='conv5', padding='same', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', name='dense1', kernel_initializer='he_normal'))
model.add(Dropout(dr))
model.add(Dense(256, activation='relu', name='dense2', kernel_initializer='he_normal'))
model.add(Dropout(dr))
model.add(Dense(classes, name='dense3', kernel_initializer='he_normal'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

nb_epoch = 200     # number of epochs to train on
batch_size = 512  # training batch size

print('----------------------Model Set Beginning Training----------------------')

filepath = 'trained_weights/Model_E_wts_data_dost.h5'

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    #show_accuracy=False,
#     verbose=2,
    validation_data=(X_val, Y_val),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)

print("Successfully trained")

test_Y_hat = model.predict(X_test, batch_size=batch_size)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)

print('--------------------------saving confusion matrix---------------------')

classes=mods
l=len(snrs)
conf_all = {}
#snrs1=np.unique(np.array(lbl)[:,1])
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = np.argmax(Y_test[i,:])//l
    k = np.argmax(test_Y_hat[i,:])//l
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = (conf[i,:] / np.sum(conf[i,:]))
cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
overall_acc = 1.0*cor/(cor+ncor)
conf_all['all']=conf
print(np.sum(conf))
print("Overall Accuracy: ", overall_acc)
#plot_confusion_matrix(confnorm, labels=classes)
#plt.savefig('OVERALL_Confusion_Matrix_Model_C_90classes_DOST.jpg',dpi=300,transparent=True)

print('--------------------------saving confusion matrix with SNRs---------------------')
acc = {}
for t in range(0,len(snrs)):
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,X_test.shape[0]):
        k = np.argmax(test_Y_hat[i,:])//l
        j = np.argmax(Y_test[i,:])
        j1=j//l
        if j % len(snrs) == t:
            conf[j1,k] = conf[j1,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = (conf[i,:] / np.sum(conf[i,:]))
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc[int(snrs[t])] = 1.0*cor/(cor+ncor)
    conf_all[(snrs[t])]=conf
    #print(np.sum(conf))
    #plt.figure()
    #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(int(snrs[t])))
    #plt.savefig(name,dpi=300,transparent=True)
for i in snrs:
    print("Overall Accuracy for SNR=",i," ",acc[i])
cPickle.dump(conf_all,open('results/Confusion matrix_'+time+'.dat','wb'))

print('--------------------------Program Successfully terminated------------------------')

