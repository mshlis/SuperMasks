import warnings 
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import keras
    from keras.layers import *
    from WideResNet import WideResidualNetwork
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle as pkl
import os, sys
import configparser

# read the config
config = configparser.ConfigParser()
config.read(sys.argv[1])

dataset = config['train']['dataset']
batch_size = config['train'].getint('batch_size')
epochs = config['train'].getint('epochs')
lr = config['train'].getfloat('lr')
weights = config['train']['weights']
mode = config['train']['mode']
save_dir = config['train']['save_dir']
save_name = config['train']['save_name']
save_init = config['init'].getboolean('save')
save_init_path = config['init']['save_path']

# dataset specifics
if dataset == 'cifar10':
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    def preprocess(x):
        return 2*(x.astype('float32') / 255.) - 1

elif dataset == 'mnist':
    (xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
    def preprocess(x):
        return 2*x.astype('float32') - 1
else:
    raise ValueError('dataset must be either mnist of cifar10')
    
xtr, xte = preprocess(xtr), preprocess(xte)
input_shape = tuple(list(xtr.shape)[1:])
if len(input_shape) == 2:
    input_shape = input_shape + (1,)
    xtr, xte = xtr[...,np.newaxis], xte[...,np.newaxis]
classes = ytr.max()+1

# help fctn for choosing to train mask or weights
def flip(model):
    for layer in model.layers:
        if hasattr(layer, 'flip'):
            layer.flip()

# load model + weights
network = WideResidualNetwork(input_shape=input_shape,
                              weights=None,
                              mask=True,
                              classes=classes)
if mode == 'weights':
    flip(network)
elif mode != 'mask':
    raise ValueError('mode must either be weights or mask')

if weights != 'None':
    network.load_weights(weights)
    
if save_init:
    network.save_weights(save_init_path)

# setup data generator
tr_gen = ImageDataGenerator(horizontal_flip=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2)

tr_gen.fit(xtr)
tr_gen = tr_gen.flow(xtr, ytr, batch_size=batch_size)

te_gen = ImageDataGenerator()
te_gen.fit(xte)
te_gen = te_gen.flow(xte, yte, batch_size=batch_size)


# training
save_path = os.path.join(save_dir, save_name+'.h5')
mc = ModelCheckpoint(save_path, save_best_only=True, monitor='val_acc', verbose=1)
rlop = ReduceLROnPlateau(monitor='acc', patience=2, factor=.1, verbose=1)
callbacks = [rlop, mc]

network.compile(keras.optimizers.Adam(lr), 
                'sparse_categorical_crossentropy', 
                metrics=['acc'])

history = network.fit_generator(tr_gen,
                                validation_data=te_gen,
                                epochs=epochs,
                                callbacks=callbacks,
                                verbose=1)

save_path = os.path.join(save_dir, save_name+'.pkl')
pkl.dump(history, open(save_path, 'wb'))