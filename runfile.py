# runfile
import keras
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import math
import os

from CifarGenerator import CifarGen
from DenseNetModel import densenet

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# file params
num_classes = 10
dataname = 'cifar{}'.format(num_classes)
path = '../database/{}/'.format(dataname)
checkpoint_path = './DenseNet_check_point.h5'
input_shape = (32, 32, 3,)

# training params
resume = False
epochs = 300
L = 40
k = 12
val_size = 0.1
batch_size = 64
lr = 0.001
w_decay = 0.0001

train_steps = math.ceil(50000 * (1 - val_size) / batch_size)
valid_steps = math.ceil(50000 * val_size / batch_size)

gen = CifarGen(path, batch_size, num_classes)
model = densenet(input_shape, num_classes, w_decay, L, k)
model_checkpoint = ModelCheckpoint(checkpoint_path, 'val_acc', 1, True, True)
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
print('data in total: ', x_train.shape[0])

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=val_size)
print('train data shape is:', X_train.shape[0])
print('validation data shape is:', X_val.shape[0])

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(gen.train_gen(X_train, Y_train),
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callback,
                    validation_data=gen.valid_gen(X_val, Y_val),
                    validation_steps=valid_steps)