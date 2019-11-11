# model file.
# Using for cifar10, L=100, k=12.
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.initializers import glorot_normal

from math import ceil

ini_seed = 1


def bn_relu(input):
    bn = BatchNormalization(axis=3)(input)
    act = Activation(activation='relu')(bn)
    return act


def bn_relu_conv(filters, trans_filters, kernel, w_decay):
    def f(input):
        br = bn_relu(input)
        conv = Conv2D(filters, kernel,
                      padding='same',
                      kernel_regularizer=l2(w_decay),
                      kernel_initializer=he_normal(),
                      )(br)
        return conv
    return f


def dense_block(input, repetitions, k, trans_filters, w_decay):
    dense = input
    for i in range(repetitions):
        conv = bn_relu_conv(k, trans_filters,(3, 3), w_decay)(dense)
        dense = Concatenate(axis=3)([dense, conv])
        trans_filters += k
    return dense, trans_filters

def dense_block_bc(input, repetitions, k, trans_filters, w_decay):
    dense = input
    rep = ceil(repetitions / 2)
    for i in range(rep):
        conv1 = bn_relu_conv(4*k, trans_filters, (1, 1), w_decay)(dense)
        conv2 = bn_relu_conv(k, trans_filters, (3, 3), w_decay)(conv1)
        dense = Concatenate(axis=3)([dense, conv2])
        trans_filters += k
    return dense, trans_filters


def transition_layer(filters, w_decay):
    def f(input):
        bn = BatchNormalization(axis=3)(input)
        conv = Conv2D(filters, (1, 1),
                      padding='same',
                      kernel_regularizer=l2(w_decay),
                      kernel_initializer=he_normal())(bn)
        pool = AvgPool2D((2, 2), strides=2, padding='same')(conv)
        return pool
    return f


def densenet(input_shape, num_classes, w_decay=0.0001, L=100, k=12):
    net = {}
    net['input'] = Input(input_shape)
    repetitions = ceil((L - 4) / 3)
    tran_filters = 16

    # layer 1
    net['conv1'] = Conv2D(16, 1,
                          padding='same',
                          kernel_regularizer=l2(w_decay),
                          kernel_initializer=he_normal(),
                          )(net['input'])

    # layer 2-33, DenseBlock 1
    net['dense1'], tran_filters= dense_block(net['conv1'], repetitions, k, tran_filters, w_decay)

    # layer  34, transition layer 1
    net['trans1'] = transition_layer(tran_filters, w_decay)(net['dense1'])

    # layer 35-66, DenseBlock 2
    net['dense2'], tran_filters= dense_block(net['trans1'], repetitions, k, tran_filters, w_decay)

    # layer 67, transition layer 2
    net['trans2'] = transition_layer(tran_filters, w_decay)(net['dense2'])

    # layer 68-99, DenseBlock 3
    net['dense3'], tran_filters= dense_block(net['trans2'], repetitions, k, tran_filters, w_decay)

    # layer 100, FC layer
    shape = K.int_shape(net['dense3'])
    net['bn'] = BatchNormalization(axis=3)(net['dense3'])
    net['act'] = Activation('relu')(net['bn'])
    net['global_avg'] = AvgPool2D((shape[1], shape[2]), strides=1)(net['act'])
    net['flat'] = Flatten()(net['global_avg'])
    net['output'] = Dense(num_classes,
                          activation='softmax',
                          kernel_initializer=glorot_normal(),
                          )(net['flat'])

    model = Model(net['input'], net['output'])
    # model.summary()
    return model


def densenet_bc(input_shape, num_classes, w_decay=0.0001, L=190, k=40):
    net = {}
    net['input'] = Input(input_shape)
    repetitions = ceil((L - 4) / 3)
    tran_filters = 16

    # layer 1
    net['conv1'] = Conv2D(16, 1,
                          padding='same',
                          kernel_regularizer=l2(w_decay),
                          kernel_initializer=he_normal(),
                          )(net['input'])

    # layer 2-33, DenseBlock 1
    net['dense1'], tran_filters= dense_block_bc(net['conv1'], repetitions, k, tran_filters, w_decay)

    # layer  34, transition layer 1
    net['trans1'] = transition_layer(tran_filters, w_decay)(net['dense1'])

    # layer 35-66, DenseBlock 2
    net['dense2'], tran_filters= dense_block_bc(net['trans1'], repetitions, k, tran_filters, w_decay)

    # layer 67, transition layer 2
    net['trans2'] = transition_layer(tran_filters, w_decay)(net['dense2'])

    # layer 68-99, DenseBlock 3
    net['dense3'], tran_filters= dense_block_bc(net['trans2'], repetitions, k, tran_filters, w_decay)

    # layer 100, FC layer
    shape = K.int_shape(net['dense3'])
    net['bn'] = BatchNormalization(axis=3)(net['dense3'])
    net['act'] = Activation('relu')(net['bn'])
    net['global_avg'] = AvgPool2D((shape[1], shape[2]), strides=1)(net['act'])
    net['flat'] = Flatten()(net['global_avg'])
    net['output'] = Dense(num_classes,
                          activation='softmax',
                          kernel_initializer=glorot_normal()
                          )(net['flat'])

    model = Model(net['input'], net['output'])
    # model.summary()
    return model