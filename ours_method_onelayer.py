from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import os
from models import *
from A_ConvNets import *
from othermodels import all_model
from read_images import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA


mode='other_test' #test train or others
test_numbers=10#the times of test
part_or_all='part' #all or part
total=5# the samples of training  
suiji=False  #mode name
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
    x_train,y_train,x_test, y_test=get_alldata(num_classes=num_classes,img_size=img_size) #第一遍取时遍历所有的样本文件夹，然后根据num_classes取前几个文件夹作为训练及测试
    if part_or_all!='all':
        x_train,y_train=get_partdata(x_train,y_train,num_classes=num_classes,total=total,mode='train',suiji=suiji)
    # x_train=x_train[:,14:114,14:114,:]
    # x_test=x_test1[:,14:114,14:114,:]
    #our_net
    model_path='model/deep_model_'+part_or_all+str(total)+'.h5'#+'_'+str(num_classes) +'_'+suiji_or_chazhi
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
        #Aconv_net
        # model_path_aconv='model_aconv/deep_model_'+part_or_all+str(total)+'.h5'#+'_'+str(num_classes) +'_'+suiji_or_chazhi
        # model_aconv=AconvNet(model_path_aconv,num_classes=num_classes)
        result_total4=[]
        result_total5=[]
        result_total6=[]
        result_total7=[]
        result_total=[]
        y_train1=np.argmax(y_train,axis=1)
        y_test1=np.argmax(y_test,axis=1)
        for i in range(test_numbers):
            dense1_layer_model =Model(inputs=model.input,outputs=model.get_layer('layer3').output)  
            fx_test3 = dense1_layer_model.predict(x_test)
            fx_train3 = dense1_layer_model.predict(x_train)
            fx_test3=change_shape(fx_test3,type='no')
            fx_train3=change_shape(fx_train3,type='no')
            pca=PCA(n_components=128)
            pca.fit(fx_train3)
            fx_tr3=pca.transform(fx_train3)
            fx_te3=pca.transform(fx_test3)
            result=all_model(fx_tr3,fx_te3,y_train1,y_test1)
            result_total4.append(result)
            dense1_layer_model =Model(inputs=model.input,outputs=model.get_layer('layer4').output)  
            fx_test4 = dense1_layer_model.predict(x_test)
            fx_train4 = dense1_layer_model.predict(x_train)
            fx_test4=change_shape(fx_test4,type='no')
            fx_train4=change_shape(fx_train4,type='no')
            pca=PCA(n_components=86)
            pca.fit(fx_train4)
            fx_tr4=pca.transform(fx_train4)
            fx_te4=pca.transform(fx_test4)      
            result=all_model(fx_tr4,fx_te4,y_train1,y_test1)
            result_total5.append(result)
            dense1_layer_model =Model(inputs=model.input,outputs=model.get_layer('dense1_out').output)  
            fx_test_dence1 = dense1_layer_model.predict(x_test)
            fx_train_dence1 = dense1_layer_model.predict(x_train)
            result=all_model(fx_train_dence1,fx_test_dence1,y_train1,y_test1)
            result_total6.append(result)
            dense1_layer_model_2 =Model(inputs=model.input,outputs=model.get_layer('soft_out').output)  
            fx_test_out = dense1_layer_model_2.predict(x_test)
            fx_train_out = dense1_layer_model_2.predict(x_train)
            result=all_model(fx_train_out,fx_test_out,y_train1,y_test1)
            result_total7.append(result)
            
            fx_train=np.concatenate((fx_tr3,fx_tr4,fx_train_dence1,fx_train_out),axis=1)  
            fx_test=np.concatenate((fx_te3,fx_te4,fx_test_dence1,fx_test_out),axis=1) 
            result=all_model(fx_train,fx_test,y_train1,y_test1)
            result_total.append(result)

            # print(i,'Rotation Forest:',result[0],'Adaboost RF:',result[1])
        s1=model.evaluate(x_test,y_test)
        # x_test=x_test1[:,20:108,20:108,:]
        # s2=model_aconv.evaluate(x_test,y_test)
        result_total=np.array(result_total)
        d=np.max(result_total,0)
        print(str(total)+' samples/one_category test_acc:','Our_cnn:',s1[1],'Rotation Forest:',d[0],'Adaboost RF:',d[1])
        result_total=np.array(result_total4)
        d=np.max(result_total,0)
        print(str(total)+' samples/layer4 test_acc:','Rotation Forest:',d[0],'Adaboost RF:',d[1])
        result_total=np.array(result_total5)
        d=np.max(result_total,0)
        print(str(total)+' samples/layer5 test_acc:','Rotation Forest:',d[0],'Adaboost RF:',d[1])
        result_total=np.array(result_total6)
        d=np.max(result_total,0)
        print(str(total)+' samples/layer6 test_acc:','Rotation Forest:',d[0],'Adaboost RF:',d[1])
        result_total=np.array(result_total7)
        d=np.max(result_total,0)
        print(str(total)+' samples/layer7 test_acc:','Rotation Forest:',d[0],'Adaboost RF:',d[1])
        




