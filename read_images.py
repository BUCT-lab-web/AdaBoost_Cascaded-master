import numpy as np
import os
from extra import utils
from sklearn.utils import check_random_state
import keras
def get_alldata(num_classes,img_size):
    index=0
    if os.path.exists('tmp_data/all_data/x_train.npy'):
        x_train=np.load('tmp_data/all_data/x_train.npy')
        index=index+1
    if os.path.exists('tmp_data/all_data/y_train.npy'):
        y_train=np.load('tmp_data/all_data/y_train.npy')
        index=index+1
    if os.path.exists('tmp_data/all_data/x_test.npy'):
        x_test=np.load('tmp_data/all_data/x_test.npy')
        index=index+1
    if os.path.exists('tmp_data/all_data/y_test.npy'):
        y_test=np.load('tmp_data/all_data/y_test.npy')
        index=index+1
    if index!=4:
        test = utils.load_img('datasets/class10/test', num_classes=num_classes,resize=img_size)
        train = utils.load_img('datasets/class10/train', type1='train',num_classes=num_classes,resize=img_size)

        x_train = train['data'].reshape(train['data'].shape[0], train['data'].shape[1], train['data'].shape[2], 1)
        y_train = train['target']
        y_train = keras.utils.to_categorical(y_train, num_classes)

        x_test = test['data'].reshape(test['data'].shape[0], test['data'].shape[1], test['data'].shape[2], 1)
        y_test = test['target']
        y_test = keras.utils.to_categorical(y_test, num_classes)

        np.save('tmp_data/all_data/x_train.npy',x_train)
        np.save('tmp_data/all_data/y_train.npy',y_train)

        np.save('tmp_data/all_data/x_test.npy',x_test)
        np.save('tmp_data/all_data/y_test.npy',y_test)
    return x_train,y_train,x_test, y_test


def get_partdata(x_train,y_train,num_classes=10, total=50,mode='train',suiji=False):
    y=np.argmax(y_train,axis=1)
    if suiji:
        if os.path.exists('savemat/x_train'+str(total)+'.npy') and os.path.exists('savemat/y_train'+str(total)+'.npy'):
            x_train=np.load('savemat/x_train'+str(total)+'.npy')
            y_train=np.load('savemat/y_train'+str(total)+'.npy')
            return x_train,y_train
        tmpi=[]
        length=0
        for i in range(num_classes):
            d1=np.where(y==i)
            tmp=d1[0].shape[0]

            random_state = check_random_state(0)
            index1 = np.arange(total)
            random_state.shuffle(index1)
            res=d1[0][index1]
            length+=len(index1)
            tmpi.append(res)
    else:
        tmpi=[]
        length=0
        for i in range(num_classes):
            d1=np.where(y==i)
            tmp=d1[0].shape[0]
            index1=[i for i in range(0,tmp,int(tmp/total)+1)]
            res=d1[0][index1]
            length+=len(index1)
            tmpi.append(res)
    x_train_re=np.zeros((length,x_train.shape[1],x_train.shape[2],x_train.shape[3]))     
    y_train_re=np.zeros((length,y_train.shape[1]))
    start=0
    for i in range(num_classes):
        end=start+tmpi[i].shape[0]
        x_train_re[start:end,:,:,:]=x_train[tmpi[i],:,:,:]
        y_train_re[start:end,:]=y_train[tmpi[i],:]
        start+=tmpi[i].shape[0]

    random_state = check_random_state(0)
    indices = np.arange(length)
    random_state.shuffle(indices)
    x_train = x_train_re[indices,:,:,:]
    y_train = y_train_re[indices,:]
    if suiji:
        np.save('savemat/x_train'+str(total)+'.npy',x_train)
        np.save('savemat/y_train'+str(total)+'.npy',y_train)
    return x_train,y_train














