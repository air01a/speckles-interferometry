
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation, Flatten,Dense,LeakyReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2


# define conv_factory: batch normalization + ReLU + Conv2D + Dropout (optional)



# define model U-net modified with dense block
def get_model_deep_speckle():
    inputs = Input((256, 256, 2))
    conv1 = Conv2D(128, 3, padding='same', strides = 2,activation='relu')(inputs)
    #x=LeakyReLU(.2)(conv1)
    conv2 =  Conv2D(128, 3,  padding='same',  strides = 2,activation='relu')(conv1)
    #x=LeakyReLU(.2)(conv2)
    conv3 =  Conv2D(64, 3,  padding='same',  strides = 2)(conv2)
    #x=LeakyReLU(.2)(conv3)
    conv4 =  Conv2D(32, 3,  padding='same',  strides = 2)(conv3)
    #x=LeakyReLU(.2)(conv4)
    conv5 =  Conv2D(16, 3,  padding='same',  strides = 2)(conv4)
    conv6 =  Conv2D(8, 3,  padding='same')(conv5)
    f6 = Flatten()(conv6)
    print("flatten : ", f6.shape)
    d6 = Dense(64,  activation='tanh')(f6)
    d7 = Dense(64,  activation='tanh')(d6)
    d8 = Dense(64,  activation='tanh')(d7)
    d9 = Dense(64,  activation='tanh')(d8)

    d10 = Dense(4,  activation='sigmoid')(d9)
    model = Model(inputs=inputs, outputs=d10)

    return model
