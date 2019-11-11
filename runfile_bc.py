# runfile
import keras
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import os
from math import ceil

from CifarGenerator import CifarGen
from DenseNetModel import densenet_bc

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# model params
L = 190
k = 40

# file params
num_classes = 100
dataname = 'cifar{}'.format(num_classes)
path = '../database/{}/'.format(dataname)
checkpoint_path = './History/CheckPoint_HKM_normal_bc _L{}_K{}.h5'.format(L, k)
history_path = './History/History_HKM_normal_bc_L{}_K{}.txt'.format(L, k)

input_shape = (32, 32, 3,)

# training params
resume = False
epochs = 200
val_size = 0.1
batch_size = 32
lr = 0.001
w_decay = 0.0001

train_steps = ceil(50000 * (1 - val_size) / batch_size)
valid_steps = ceil(50000 * val_size / batch_size)

gen = CifarGen(path, batch_size, num_classes)
model = densenet_bc(input_shape, num_classes, w_decay, L, k)
model_checkpoint = ModelCheckpoint(checkpoint_path, 'val_accuracy', 1, True, True)
if resume:
    model.load_weights(checkpoint_path)
opt = keras.optimizers.Adam(lr)


def lr_reducer(epoch):
    rate_1 = epochs * 0.5
    rate_2 = epochs * 0.75
    if epoch > rate_1:
        if epoch > rate_2:
            new_lr = lr * 0.1 * 0.1
        else:
            new_lr = lr * 0.1
    else:
        new_lr = lr
    return new_lr


callback=[keras.callbacks.LearningRateScheduler(lr_reducer, verbose=1), model_checkpoint]

x_train, y_train = gen.train_data()
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=val_size)
print('train data shape is:', X_train.shape[0])
print('validation data shape is:', X_val.shape[0])

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(gen.train_gen(X_train, Y_train),
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callback,
                    validation_data=gen.valid_gen(X_val, Y_val),
                    validation_steps=valid_steps)

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']

np_train_acc = np.array(train_acc).reshape((-1, 1))
np_valid_acc = np.array(valid_acc).reshape((-1, 1))
np_out = np.concatenate([np_train_acc, np_valid_acc], axis=1)
np.savetxt(history_path, np_out, fmt='%.7f')
