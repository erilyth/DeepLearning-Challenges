# Text Based Chatbot

A chatbot that takes a small story and a query as an input and predicts a possible answer to the query. After training, a user can give similar stories and queries that the chatbot was trained on it would give accurate results. 

## Overview

This is the code for [this](https://youtu.be/t5qgjJIBy9g) by Siraj. This code trains an end-to-end memory network, since there doesn't yet exist a 'dynamic' memory network implementation in Keras. For an example of a 'dynamic' memory network see [this](https://github.com/ethancaballero/Improved-Dynamic-Memory-Networks-DMN-plus) repository.

(Trained models have been included in `trained_models/`)

## Dependencies

* tensorflow (https://www.tensorflow.org/install/)
* keras
* functools
* tarfile
* re

## Results/Observations

|Layers 						|Dropouts			|Batch-size	| Epochs 	|Results	|
| ----------------------------- | ----------------- | --------- | --------- | --------- |
|LSTM(32)						|(0.3)				|32			|100		|95.8%		|
|LSTM(64)						|(0.3)				|32			|100		|96.6%		|
|LSTM(32), LSTM(32)				|(0.5, 0.5)			|32			|100		|92.3%		|
|LSTM(32), LSTM(32)				|(0.5, 0.5)			|32			|200		|96.9%		|
|GRU(32)						|(0.3)				|32			|100		|86.3%		|
|GRU(64)						|(0.3)				|32			|100		|94.3%		|
|GRU(32), GRU(32)				|(0.5, 0.5)			|64			|100		|65.2%		|
|GRU(32), GRU(32)				|(0.5, 0.5)			|64			|300		|93.5%		|

* The models with two or more layers required more training since there are more parameters that need to be set, but then have greater accuracies than the other models once trained completely.
* Overall, LSTM based models performed better than GRU based models for this task.
* The dataset used here is `babi-tasks-v1-2`, its a relatively small dataset but a great dataset nonetheless.

## Usage

* Set the parameters as required in memorynetwork.py (train_model, train_epochs, load_model etc)
* Run `python memorynetwork.py`
* If you set `test_qualitative = 1` then a few sample test queries and the predictions are shown
* If you set `user_questions = 1` then the user is prompted for input once the model is loaded so they can interact with the chatbot

## Credits

Credits for the base code go to the creator of Keras, [fchollet](https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py)
