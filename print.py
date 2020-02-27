import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical


def plot_acc(history, title="Model Accuracy"):
    """Imprime una gráfica mostrando la accuracy por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
def plot_loss(history, title="Model Loss"):
    """Imprime una gráfica mostrando la pérdida por epoch obtenida en un entrenamiento"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
    
def visualize_example(x):
    plt.figure()
    plt.imshow(x)
    plt.colorbar()
    plt.grid(False)
    plt.show()



def evaluate(model, history, x_val, y_val):
    print('model evaluate')
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    print(title)
    print(' loss {}, acc: {}\n'.format(loss, acc))
    plot_acc(history, "acc model")
    plot_loss(history, "loss model")
    print(time.strftime("%I:%M:%S"))

 