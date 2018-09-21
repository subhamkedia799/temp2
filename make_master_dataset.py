import pickle
import numpy as np
Xd = pickle.load(open('dataset/RML2016.10b.dat','rb'), encoding='latin')
Xa= pickle.load(open('dataset/RML2016.10a_dict.pkl','rb'), encoding='latin')
Xc= pickle.load(open('dataset/2016.04C.multisnr.pkl','rb'), encoding='latin')
new_Xd={}
for keys in Xd.keys():
        new_Xd[keys] = np.concatenate((Xd[keys],Xa[keys],Xc[keys]),axis=0)

pickle.dump(new_Xd,open('dataset/master_dataset.dat','wb'))