import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model,Input
import os
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
def CNNModel(path,num_classes,input_shape):
    x_in=Input(input_shape)
    x=Conv2D(32,kernel_size=(3,3))(x_in)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x1=Conv2D(32,kernel_size=(3,3), activation='relu',name='layer1')(x)
    x=MaxPooling2D(pool_size=(2,2))(x1)
    x2=Conv2D(32,kernel_size=(3,3), activation='relu',name='layer2')(x)
    x=MaxPooling2D(pool_size=(2,2))(x2)
    x3=Conv2D(64,kernel_size=(3,3), activation='relu',name='layer3')(x)
    x=MaxPooling2D(pool_size=(2,2))(x3)
    x4=Conv2D(64,kernel_size=(3,3), activation='relu',name='layer4')(x)
    x=MaxPooling2D(pool_size=(2,2))(x4)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x5=Dense(128, activation='relu',name='dense1_out')(x)
    x=Dropout(0.5)(x5)
    x1=Flatten()(x1)
    x2=Flatten()(x2)
    x3=Flatten()(x3)
    x4=Flatten()(x4)
    x=Concatenate()([x1,x2,x3,x4,x5])
    out=Dense(num_classes,activation='softmax',name='soft_out')(x)
    model=Model(input=x_in,output=out)
    # model.summary()
    # plot_model(model, to_file='model.png')
    if os.path.isfile(path):
        model.load_weights(path)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

CNNModel('.',10,(128,128,1))








