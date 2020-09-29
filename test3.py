from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import os
from models_concate import *
from A_ConvNets import *
from othermodels import all_model
from read_images import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA


mode='test'  
test_numbers=10 
part_or_all='part'  
total=5 
suiji=False  
input_shape = (128, 128, 1)

num_classes =10  
img_size = (128, 128)
epochs =100
batch_size =16

def change_shape(data,type='min'):
    if type=='max':
        fx=np.max(data,axis=-1)
    elif type=='average':
        fx=np.average(data,axis=-1)
    elif type=='no':
        fx=data
    else:
        fx=np.min(data,axis=-1)
    fd=np.reshape(fx,(fx.shape[0],-1))
    return fd

datasets=[5,10,20,30,40,50]

for i in datasets:
    total=i
    x_train,y_train,x_test1, y_test=get_alldata(num_classes=num_classes,img_size=img_size) 
    if part_or_all!='all':
        x_train,y_train=get_partdata(x_train,y_train,num_classes=num_classes,total=total,mode='train',suiji=suiji)
    # x_train=x_train[:,14:114,14:114,:]
    x_test=x_test1
    #our_net
    model_path='model_concate/deep_model_'+part_or_all+str(total)+'.h5'#+'_'+str(num_classes) +'_'+suiji_or_chazhi
    model=CNNModel(model_path,num_classes=num_classes,input_shape=(x_train.shape[1],x_train.shape[1],1))
   
    if mode=='train':
        checkpoint = ModelCheckpoint(filepath=model_path,monitor='val_acc',mode='auto' ,save_best_only='True')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),callbacks=[checkpoint])
    elif mode=='test':
        s=model.evaluate(x_test,y_test)
        print('CNN:',s)
    else:
        print('none')





