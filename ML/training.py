from __future__ import print_function
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from model import get_model_deep_speckle
from model2 import get_model_deep_speckle
from matplotlib import pyplot as plt
#from loss_function_new import total_variation_balanced_cross_entropy, balanced_cross_entropy, gaussian_loss
#from parameter import save_path, num_epochs, batch_size, save_period, lr_rate

from os import listdir
from os.path import splitext, isfile, isdir, join
from sklearn.model_selection import train_test_split

# Split the data


#plt.switch_backend('agg')


proj_name = 'binarystar'
show_groundtruth_flag = False
main_dir = '.'

maindir = "simulated/"
save_path='.'
num_epochs=80
batch_size=4
save_period=10
lr_rate=0.01

def train_and_predict():
    X = []
    print('-' * 30)
    print('Loading  training data...')
    print('-' * 30)

    print('-' * 30)
    print('create validation data...')
    print('-' * 30)

    files = [f for f in listdir(maindir) if f.endswith('.npy')]

    # Trier les fichiers en entrées (x) et sorties (y)
    X = sorted([f for f in files if f.startswith('x')])
    Y = sorted([f for f in files if f.startswith('c')])

    # Assurez-vous que les paires x et y correspondent
    # Ici, nous supposons que les fichiers sont correctement nommés pour que cette opération soit valide
    assert len(X) == len(Y)

    # Charger les images
    images_x = [np.load(join(maindir, f)) for f in X]


    #images_y = [np.expand_dims(np.load(join(maindir, f)),axis=-1) for f in Y]
    images_y = [np.load(join(maindir, f)) for f in Y]

    # Convertir en tableaux NumPy pour le training
    dataset_x = np.array(images_x)
    dataset_y = np.array(images_y)
    dataset_y = dataset_y 
    #plt.imshow(dataset_x[100])
    #print(dataset_y[100])
    #plt.show()
    x_train, x_valid, y_train, y_valid = train_test_split(dataset_x, dataset_y, test_size=0.2, shuffle= True)
    print(x_train.shape, y_train.shape)
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train_load, y_train_load, test_size=0.02, shuffle= True)
    # np.save('../data/x_valid.npy',x_valid)
    # np.save('../data/y_valid.npy',y_valid)
    # np.save('../data/x_train_new.npy',x_train)
    # np.save('../data/y_train_new.npy',y_train)


    #x_train = np.load('../data/x_train.npy')
    #print(x_train.shape)
    #print(y_train.shape)
    #y_train = np.load('../data/y_train.npy')
    ##x_valid = np.load('../data/x_vali.npy')
    #y_valid = np.load('../data/y_vali.npy')
    #print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_model_deep_speckle()
    model.load_weights('.binarystar.80.hdf5')

    #model.load_weights('save/lr4/MNIST_x2.60.hdf5')
    model.compile(optimizer=Adam(lr=lr_rate), loss='MeanSquaredError')
    model_checkpoint = ModelCheckpoint(save_path+proj_name+'.{epoch:02d}.hdf5', monitor='loss', verbose=2, save_best_only=False,
                                       period=save_period)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, shuffle=True,
              callbacks=[model_checkpoint], validation_data = (x_valid, y_valid))


    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('total_loss_4.png')
    plt.close()


if __name__ == '__main__':
    train_and_predict()