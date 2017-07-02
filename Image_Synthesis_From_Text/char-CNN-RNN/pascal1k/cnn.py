import keras
import random
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

batch_size = 16
num_classes = 20
epochs = 1000

# input image dimensions
img_rows, img_cols = 128, 256

train_data = []
train_labels = []

test_data = []
test_labels = []

filelist = open('filelist.txt', 'r')

for file in filelist.readlines():
  fname, classid = file.split(' ')
  classid = int(classid) - 1 # Make them zero indexed
  imgfile = Image.open(fname)
  img = imgfile.resize((img_rows, img_cols), Image.ANTIALIAS)  
  img = np.asarray(img)
  if random.random() >= 0.02:
    train_data.append(img)
    train_labels.append(classid)
  else:
    test_data.append(img)
    test_labels.append(classid)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

#print train_data.shape, train_labels.shape

if K.image_data_format() == 'channels_first':
    train_data = train_data.reshape(train_data.shape[0], 3, img_rows, img_cols)
    test_data = test_data.reshape(test_data.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 3)
    test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255
print('train_data shape:', train_data.shape)
print(train_data.shape[0], 'train samples')
print(test_data.shape[0], 'test samples')

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

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
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

checkpointer = ModelCheckpoint(filepath="cnn_weights.hdf5", verbose=1)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_labels),
          callbacks=[checkpointer])

score = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
