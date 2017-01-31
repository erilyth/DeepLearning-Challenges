from __future__ import division, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor

from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

load_model = 0
save_model = 0

# Select only the two columns we require. Game title and its corresponding emotion
dataframe = pd.read_csv('ign.csv').ix[:, 1:3]
# Fill null values with empty strings
dataframe.fillna(value='', inplace=True)

# print(dataframe.score_phrase.value_counts())

# Extract the required columns for inputs and outputs
totalX = dataframe.title
totalY = dataframe.score_phrase

# Convert the strings in the input into integers corresponding to the dictionary positions
# Data is automatically padded so we need to pad_sequences manually
vocab_proc = VocabularyProcessor(15)
totalX = np.array(list(vocab_proc.fit_transform(totalX)))

# We will have 11 classes in total for prediction, indices from 0 to 10
vocab_proc2 = VocabularyProcessor(1)
totalY = np.array(list(vocab_proc2.fit_transform(totalY))) - 1
# Convert the indices into 11 dimensional vectors
totalY = to_categorical(totalY, nb_classes=11)

# Split into training and testing data
trainX, testX, trainY, testY = train_test_split(totalX, totalY, test_size=0.1)

# Build the network for classification
# Each input has length of 15
net = tflearn.input_data([None, 15])
# The 15 input word integers are then casted out into 256 dimensions each creating a word embedding.
# We assume the dictionary has 10000 words maximum
net = tflearn.embedding(net, input_dim=10000, output_dim=256)
# Each input would have a size of 15x256 and each of these 256 sized vectors are fed into the LSTM layer one at a time.
# All the intermediate outputs are collected and then passed on to the second LSTM layer.
net = tflearn.gru(net, 256, dropout=0.9, return_seq=True)
# Using the intermediate outputs, we pass them to another LSTM layer and collect the final output only this time
net = tflearn.gru(net, 256, dropout=0.9)
# The output is then sent to a fully connected layer that would give us our final 11 classes
net = tflearn.fully_connected(net, 11, activation='softmax')
# We use the adam optimizer instead of standard SGD since it converges much faster
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
						 loss='categorical_crossentropy')

# Train the network
model = tflearn.DNN(net, tensorboard_verbose=0)

if load_model == 1:
	model.load('gamemodel.tfl')

model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32, n_epoch=20)

if save_model == 1:
	model.save('gamemodel.tfl')
	print "Saved model!"