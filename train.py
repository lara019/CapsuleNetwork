import numpy as np
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from print import evaluate
from keras.datasets import cifar10
from keras.models import model_from_yaml

import pickle
import os

from sklearn.model_selection import train_test_split
from scipy import interpolate
from skimage.transform import resize
from print import visualize_example
import other_utils
from time import gmtime, strftime
from matplotlib import pyplot
import keras
from keras.preprocessing.image import ImageDataGenerator

import time
import datetime

batch_size = 100
epochs = 200
shift_fraction = 0.1
save_dir = './result/'
lr = 0.001
lam_recon = 0.392

# callbacks
tb = callbacks.TensorBoard(log_dir=save_dir + 'tensorboard-logs',
                               batch_size=batch_size, histogram_freq=0)
checkpoint = callbacks.ModelCheckpoint(save_dir + 'weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                       save_best_only=True, save_weights_only=True, verbose=1)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (0.95 ** epoch))
early_stopper = callbacks.EarlyStopping(min_delta=0.001, patience=10)



def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))



def train(model, data, nombre):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # compile the model
    print('model compile')
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., lam_recon],
                  metrics={'capsnet': 'accuracy'})

    model_DA = model

    time1= time.clock()
    print(time1)
    nombre1 = nombre + 'sinDA'
    model_sin, history_sin = train_sinDA(model, data, nombre1)
    tiempo_sin = time.clock() - time1
    time1 = time.clock()
    print(time1)
    save(model_sin, nombre1)
    
    
    nombre1 = nombre + 'DA'
    model_DA, history_DA = train_DA(model_DA, data, nombre1)
    tiempo_con = time.clock() - time1
    print("tiempo_sin: ", tiempo_sin, "tiempo_con: ", tiempo_con)
    print(time.time())
    save(model_DA, nombre1)
    return model_DA, history_DA, model_sin, history_sin

def train_sinDA(model, data, nombre):

    # Training without data augmentation:
    print('model fit: ' + nombre)
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    log = callbacks.CSVLogger(save_dir + nombre + '.csv')

    history = model.fit([x_train, y_train], [y_train, x_train], batch_size = batch_size, epochs=epochs,
    validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay, early_stopper])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    return model, history


def save(model, nombre):
    model_yaml = model.to_yaml()
    with open(save_dir  + nombre + '.yaml', "w") as yaml_file:
        yaml_file.write(save_dir  + model_yaml)
    # serialize weights to HDF5
    model.save_weights(save_dir + nombre)
    print('Trained model saved to', nombre)


def train_DA(model, data, nombre):
    (x_train, y_train), (x_test, y_test) = data
    log = callbacks.CSVLogger(save_dir  + nombre + '.csv')
    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
        height_shift_range=shift_fraction,
        horizontal_flip=True)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    print('model fit_generator: ' + nombre)
    history = model.fit_generator(generator=train_generator(x_train, y_train, batch_size, shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay, early_stopper])
    return model, history
