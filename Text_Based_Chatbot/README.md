# Text Based Chatbot

A chatbot that takes a small story and a query as an input and predicts a possible answer to the query

## Overview

This is the code for [this](https://youtu.be/t5qgjJIBy9g) by Siraj. This code trains an end-to-end memory network, since there doesn't yet exist a 'dynamic' memory network implementation in Keras. For an example of a 'dynamic' memory network see [this](https://github.com/ethancaballero/Improved-Dynamic-Memory-Networks-DMN-plus) repository. 

## Dependencies

* tensorflow (https://www.tensorflow.org/install/)
* keras
* functools
* tarfile
* re

## Usage

* Set the parameters as required in memorynetwork.py (train_model, train_epochs, load_model etc)
* Run `python memorynetwork.py`

## Credits

Credits for the base code go to the creator of Keras, [fchollet](https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py)
