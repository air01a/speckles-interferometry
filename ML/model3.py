
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation, Flatten,Dense,LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.regularizers import l2


# define conv_factory: batch normalization + ReLU + Conv2D + Dropout (optional)



# define model U-net modified with dense block
def get_model_deep_speckle():
    inputs = Input((256, 256, 2))
    conv1 = Conv2D(16, 3, padding='same', activation='relu')(inputs)
    #x=LeakyReLU(.2)(conv1)
    conv2 =  Conv2D(32, 3,  padding='same',  activation='relu')(conv1)
    conv2d = Dropout(0.40)(conv2)
    #x=LeakyReLU(.2)(conv2)
    conv3 =  Conv2D(64, 3,  padding='same',  strides = 2,activation='relu')(conv2d)
    conv3d = Dropout(0.40)(conv3)

    #x=LeakyReLU(.2)(conv3)
    conv4 =  Conv2D(64, 3,  padding='same',  activation='relu')(conv3d)
    conv4d = Dropout(0.40)(conv4)

    #x=LeakyReLU(.2)(conv4)
    conv5 =  Conv2D(128, 3,  padding='same',  strides = 2,activation='relu')(conv4d)
    conv5d = Dropout(0.3)(conv5)

    conv6 =  Conv2D(256, 3,  padding='same',activation='relu', strides=2)(conv5d)
    conv6d = Dropout(0.3)(conv6)

    conv7 =  Conv2D(256, 3,  padding='same',activation='relu', strides=2)(conv6d)
    conv7d = Dropout(0.3)(conv6)
    
    
    f6 = Flatten()(conv6)
    print("flatten : ", f6.shape)
    d6 = Dense(128,  activation='tanh')(f6)
    d6d = Dropout(0.25)(d6)

    d7 = Dense(64,  activation='tanh')(d6d)
    d7d = Dropout(0.25)(d7)

    d8 = Dense(32,  activation='tanh')(d7d)
    d8d = Dropout(0.25)(d8)

    d9 = Dense(32,  activation='tanh')(d8d)
    d9d = Dropout(0.25)(d9)

    d10 = Dense(4,  activation='sigmoid')(d9d)
    model = Model(inputs=inputs, outputs=d10)
    print(model.summary())
    return model
