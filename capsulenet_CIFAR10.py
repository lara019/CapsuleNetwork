"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 100
       python CapsNet.py --epochs 100 --num_routing 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
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
from train import train


K.set_image_data_format('channels_last')

def plot(x):
    for i in range(9):
    # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(x[i])
    # show the figure
    pyplot.show()


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, 
        strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)
    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


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



    from utils import plot_log
    #plot_log(args.save_dir + '/log_Cifar10.csv', show=True)

    return model, history


def test(model, data):
    print('model predict')
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()

def resize28(arrayCompleto, w, h):
    arrayCompleto_nuevo = []
    arrayCompleto_nuevo1 = []
    imagenes = []
    for i in range(0, arrayCompleto.shape[0]):
        imagen = arrayCompleto[i]
        W, H = imagen.shape[:2]
        new_W, new_H = (w, h)
        xrange = lambda x: np.linspace(0, 1, x)
        
        f = interpolate.interp2d(xrange(W), xrange(H), imagen, kind="linear")
        new_arr = f(xrange(new_W), xrange(new_H))
        new_arr_1 = resize(new_arr, (w, h, 1))
        arrayCompleto_nuevo.append(new_arr_1)#este resize lo hacemos luego
        arrayCompleto_nuevo1.append(new_arr)

    return np.asarray(arrayCompleto_nuevo1)

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist, fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape)
    plot(x_train)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_cifar_10():

    # the data, shuffled and split between train and test sets
    batch_size = 128
    num_classes = 10
    epochs = 100
    tam = 28
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #plot(x_train)
    print(x_train.shape)
    x_train = other_utils.rgb2gray(x_train)
    x_test = other_utils.rgb2gray(x_test)
    
    x_train = resize28(x_train, 28, 28)
    x_test = resize28(x_test, 28, 28)
    plot(x_train)

    x_train = x_train.reshape(-1, tam, tam, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, tam, tam, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    data = (x_train, y_train, x_val, x_test, y_test, y_val)
    np.save('xtrainCIFAR_10.npy', data, allow_pickle=True)

    return (x_train, y_train, x_val), (x_test, y_test, y_val)

def load_CIFAR_10_fromFile( ):
    data = np.load('xtrainCIFAR_10.npy', allow_pickle=True)
    x_train, y_train, x_val, x_test, y_test, y_val = data
    return (x_train, y_train, x_val), (x_test, y_test, y_val)


if __name__ == "__main__":
    strftime("%Y-%m-%d %H:%M:%S", gmtime())
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dataAgumentation = False
    # load data
    (x_train, y_train, x_val), (x_test, y_test, y_val) = load_CIFAR_10_fromFile( )
    # create training and testing vars
    print("CIFAR shapes: x_train: ", x_train.shape, "x_test: ", x_test.shape, "x_val: ", x_val.shape)
    
    # define model
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                n_class=len(np.unique(np.argmax(y_train, 1))),
                                num_routing=args.num_routing)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        data=((x_train, y_train), (x_test, y_test))
        nombre = 'CN_cifar10'
        model, history = train(model=model, data=((x_train, y_train), (x_test, y_test)), nombre=nombre)
        
        print("Fin train.")
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        evaluate(model, history, x_val, y_val)
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test))
        
        print("Fin test.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    print("Fin fin.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
