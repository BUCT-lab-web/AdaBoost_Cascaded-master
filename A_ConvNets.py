import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import os
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
input_shape = (88, 88, 1)
def AconvNet(path,num_classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu',name='layer1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (6, 6), activation='relu',name='layer2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu',name='layer3'))
    # model.add(Dropout(0.5))
    model.add(Conv2D(num_classes, (3, 3), activation='softmax',name='layer4'))
    model.add(Flatten())
    # model.summary()
    # plot_model(model, to_file='model.png')
    if os.path.isfile(path):
        model.load_weights(path)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

# AconvNet('.',10)









