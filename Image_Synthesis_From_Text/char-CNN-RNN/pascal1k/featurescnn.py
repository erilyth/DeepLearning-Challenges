import keras
import random
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import lutorpy as lua

require("torch")

batch_size = 16
num_classes = 20
epochs = 1000

# input image dimensions
img_rows, img_cols = 128, 256

train_data = []
train_labels = []

file_names = []

filelist = open('filelist.txt', 'r')

for file in filelist.readlines():
  fname, classid = file.split(' ')
  file_names.append(fname)
  classid = int(classid) - 1 # Make them zero indexed
  imgfile = Image.open(fname)
  img = imgfile.resize((img_rows, img_cols), Image.ANTIALIAS)  
  img = np.asarray(img)
  train_data.append(img)
  train_labels.append(classid)
  
train_data = np.array(train_data)
train_labels = np.array(train_labels)

#print train_data.shape, train_labels.shape

if K.image_data_format() == 'channels_first':
    train_data = train_data.reshape(train_data.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

train_data = train_data.astype('float32')
train_data /= 255
print('train_data shape:', train_data.shape)
print(train_data.shape[0], 'train samples')

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
dense_lyr_out = Dense(1024, activation='relu')
model.add(dense_lyr_out)
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.load_weights('cnn_weights.hdf5')

dense_lyr_out_f = K.function([model.layers[0].input, K.learning_phase()], [dense_lyr_out.output,])

for data_point in range(len(train_data)):
  cur_out =  dense_lyr_out_f([[train_data[data_point],], 0])[0]
  cur_out = np.expand_dims(np.asarray(cur_out), axis=2)
  out_name = file_names[data_point] + 'features.t7'
  tensor_t = torch.fromNumpyArray(cur_out)
  torch.save(out_name, tensor_t)
