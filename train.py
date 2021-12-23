# =============================================================================
#  @misc{punnappurath2020little,
#    title={A Little Bit More: Bitplane-Wise Bit-Depth Recovery},
#    author={Abhijith Punnappurath and Michael S. Brown},
#    year={2020},
#    eprint={2005.01091},
#    archivePrefix={arXiv},
#    primaryClass={eess.IV}
#    } 
# by Abhijith Punnappurath (05/2020)
# pabhijith@eecs.yorku.ca
# https://abhijithpunnappurath.github.io
# =============================================================================

# run this to train the model

import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Add
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam
import data_generator as dg
import copy
from keras.losses import binary_crossentropy
import keras

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bitpos', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/train', type=str, help='path of train data')
parser.add_argument('--val_data', default='data/val', type=str, help='path of val data')
parser.add_argument('--quant', default=4, type=int, help='quantize to these many bits and train model to predict next bit')
parser.add_argument('--dep', default=4, type=int, help='number of residual units')
parser.add_argument('--epoch', default=30, type=int, help='number of training epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()

print(args)

save_dir = os.path.join('models','D'+str(args.dep),args.model+'_'+str(args.quant+1).zfill(2)) 

if not os.path.exists('models'):
    os.mkdir('models')
    
if not os.path.exists(os.path.join('models','D'+str(args.dep))):
    os.mkdir(os.path.join('models','D'+str(args.dep)))
    
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def DnCNN(depth,filters=64,image_channels=3, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))    
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)        
    layer_count += 1 
    
    for i in range(depth):
        if use_bnorm:
            xr = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
            layer_count += 1            
        xr = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(xr)
        layer_count += 1            
        if use_bnorm:
            xr = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(xr) 
            layer_count += 1            
        xr = Activation('relu',name = 'relu'+str(layer_count))(xr)
        layer_count += 1            
        xr = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(xr)
        layer_count += 1 
        x = Add(name = 'add' + str(layer_count))([xr, x])   # residual unit          
        layer_count += 1 
        
    if use_bnorm:
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x) 
        layer_count += 1                    
        
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',use_bias = False,padding='same',name = 'conv'+str(layer_count))(x)
    layer_count += 1
    
    x = Activation('sigmoid',name = 'sigmoid' + str(layer_count))(x)  
    model = Model(inputs=inpt, outputs=x)
    
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=15:
        lr = initial_lr
    else:
        lr = initial_lr/5 
    log('current learning rate is %2.8f' %lr)
    return lr
  

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dir=args.train_data, batch_size=args.batch_size, shuffle=True):
        'Initialization'
        print('Inside data generator')
        xs = dg.datagenerator(data_dir,batch_size)        
        print('Number of samples : ',xs.shape[0])
        print('Outside data generator')
        
        # Generate data
        ys=copy.deepcopy(xs)
        ys=ys.astype(np.float32)
        ys=np.floor(ys/2**(16-args.quant))*2**(16-args.quant)
        ys=ys/65535.0

        zs=copy.deepcopy(xs)
        zs=zs.astype(np.float32)
        zs=np.floor(zs/2**(16-args.quant-1))*2**(16-args.quant-1)
        zs=zs/65535.0
        ngt=zs-ys
        if ngt.max() != 0:
           ngt = ngt/(ngt.max())
                                
        full_x = ngt
        full_y = ys

        print('Data generation complete \n')

    
        self.batch_size = batch_size
        self.full_x = full_x
        self.full_y = full_y
        self.list_IDs = list(range(xs.shape[0]))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        batch_x = self.full_x[list_IDs_temp]
        batch_y = self.full_y[list_IDs_temp]
            
        return batch_y, batch_x
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  

    
if __name__ == '__main__':
    
    # model selection
    model = DnCNN(depth=args.dep,filters=64,image_channels=3,use_bnorm=True)
    model.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        print(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch))
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=['binary_accuracy'])
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), 
                monitor='val_metric', mode='max', verbose=1, save_weights_only=False,
                save_best_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    tensorboard_callback = TensorBoard(log_dir='./tensorboard/logs')
    
    history = model.fit_generator(DataGenerator(batch_size=args.batch_size,data_dir=args.train_data), 
                epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                validation_data=DataGenerator(batch_size=args.batch_size,data_dir=args.val_data), 
                callbacks=[checkpointer,csv_logger,lr_scheduler,tensorboard_callback])