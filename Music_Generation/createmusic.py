from mido import MidiFile, MidiTrack, Message

from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import mido

########### PROCESS MIDI FILE #############
mid = MidiFile('allegroconspirito.mid') # a Mozart piece

notes = []

time = float(0)
prev = float(0)

for msg in mid:
	### this time is in seconds, not ticks
	time += msg.time
	if not msg.is_meta:
		### only interested in piano channel
		if msg.channel == 0:
			if msg.type == 'note_on':
				# note in vector form to train on
				note = msg.bytes()
				# only interested in the note and velocity. note message is in the form of [type, note, velocity]
				note = note[1:3]
				notes.append(note)
###########################################

######## SCALE DATA TO BETWEEN -1, 1 #######
notesa = zip(*notes)[0]
notesb = zip(*notes)[1]
amin = np.min(notesa)
amax = np.max(notesa)
bmin = np.min(notesb)
bmax = np.max(notesb) 

for note in notes:
	note[0] = 2*(note[0]-((amin+amax)/2))/(amax-amin)
	note[1] = 2*(note[1]-((bmin+bmax)/2))/(bmax-bmin)

###########################################

############ CREATE DATA, LABELS ##########
X = []
Y = []
n_prev = 30
# n_prev notes to predict the (n_prev+1)th note
for i in range(len(notes)-n_prev):
	x = notes[i:i+n_prev]
	y = notes[i+n_prev]
	X.append(x)
	Y.append(y)
# save a seed to do prediction later
seed = notes[0:n_prev]
###########################################

############### BUILD MODEL ###############
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(n_prev, 2), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, input_shape=(n_prev, 2), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(64, input_shape=(n_prev, 2), return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(2))
model.add(Activation('linear'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='mse', optimizer='rmsprop')
model.fit(X, Y, 300, 20, verbose=1)
###########################################

############ MAKE PREDICTIONS #############
prediction = []
x = seed
x = np.expand_dims(x, axis=0)

for i in range(300):
	preds = model.predict(x)
	print (preds)
	x = np.squeeze(x)
	x = np.concatenate((x, preds))
	x = x[1:]
	x = np.expand_dims(x, axis=0)
	preds = np.squeeze(preds)
	prediction.append(preds)

for pred in prediction:
	pred[0] = int((pred[0]/2)*(amax-amin) + (amin+amax)/2)
	pred[1] = int((pred[1]/2)*(bmax-bmin) + (bmin+bmax)/2)
	# to reject values that will be out of range
	if pred[0] < 24:
		pred[0] = 24
	elif pred[0] > 102:
		pred[0] = 102
	if pred[1] < 0:
		pred[1] = 0
	elif pred[1] > 127:
		pred[1] = 127
###########################################

###### SAVING TRACK FROM BYTES DATA #######
mid = MidiFile()
track = MidiTrack()

t = 0
for note in prediction:
	# 147 means note_on
	note = np.asarray([147, note[0], note[1]])
	bytes = note.astype(int)
	msg = Message.from_bytes(bytes[0:3])
	t += 1
	msg.time = t
	track.append(msg)

mid.tracks.append(track)
mid.save('new_song.mid')
###########################################
