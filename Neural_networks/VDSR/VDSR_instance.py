# define VDSR architecture and train on high-res/low-res patch pairs
# patches are passed through network as tensors
# expected tensor shape (# examples ,image_size_x,image_size_y,# channels)
# "data" and "label" tensors need to be same size, including number of channels
# where "data" is interpolated low resolution imges and "label" is high
# resolution images 

#%% Load libraries
from keras.models import Model
from keras.layers import Activation
from keras.layers import Conv2D,  Input,  add
from keras.optimizers import  Adam

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
#import tensorflow.keras.backend as KerasBackend
import h5py # only need this if you using data from gthub page
import numpy as np

#%%
# standard 20 layer VDSR with zero-padding in between layers and gradient clipping
def model_instance_master(channels, loss_rate, clip_norm):
    input_img = Input((None,None,channels))
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    
    # padding = 'same' should be zero padding
    for ii in range(0,18):
        
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    
    model = Activation('relu')(model)
    model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model
    
    output_img = add([res_img, input_img])
    
    model = Model(input_img, output_img)
    
    adam = Adam(lr=loss_rate,clipnorm = clip_norm) 
    
    # loss function is an input argument but MSE seems to generally work well
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

# plot on a log scale loss and validation loss from second entry onwards
def plot_history(history_instance):
    FS = 18
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.plot(history_instance.history['loss'][1:], label='train')
    plt.plot(history_instance.history['val_loss'][1:], label='test')
    plt.yscale('log')
    plt.legend(fontsize = FS)
    plt.xlabel('Epoch',fontsize = FS)
    plt.ylabel('Loss', fontsize = FS)
    plt.xticks(fontsize = FS-4)
    plt.yticks(fontsize = FS-4)
    plt.show()
    
# only need this funciton is you use data provided to you
#otherwise load in "data" and label" images for training and validation yourself
    
def read_data_BT_SST(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        SST_INT = np.array(hf.get('SST_INT'))
        SST_GT = np.array(hf.get('SST_GT'))
        
        
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label, SST_INT, SST_GT
    
#%% load training data and labels
train_filename = "train_multi-chan_VDSR.h5"
test_filename = "test_multi-chan_VDSR.h5"
data, label, SST_unused1, SST_unused_2 = read_data_BT_SST(train_filename)
val_data, val_label, S3_SST_INT, S3_SST_GT = read_data_BT_SST(test_filename)
#%% train network
    
model_1 = []
checkpoint = ModelCheckpoint("VDSR_check_1.h5", monitor='val_loss', verbose=1, save_best_only=True,
                     save_weights_only=False, mode='min')

callbacks_list = [checkpoint]
N_channels = 3
model_1 = model_instance_master(N_channels,0.0001, 0.1)

history1 = model_1.fit(data, label, batch_size=64, validation_data=(val_data, val_label),
            callbacks=callbacks_list, epochs=10, verbose=0)

# plot loss
plot_history(history1)
    