# How_to_make_a_chatbot
This is the code for "How to Make a Chatbot - Intro to Deep Learning #12' by Siraj Raval on YouTube

## Coding challenge - Due Date, Thursday April 6 2017 at 12 PM PST

The challenge for this video is to make your own Q&A System using any type of memory network (plain, end-to-end, or dynamic). You can even use the code in this repository verbatim, just modify it so that a user can ask it questions from the command line. This will be a good way to get familiar with this type of bleeding edge deep learning model.

## Overview

This is the code for [this](https://youtu.be/t5qgjJIBy9g) video by Siraj Raval on Youtube as part of the Udacity Deep Learning Nanodegree. This code is actually an end-to-end memory network, since there doesn't yet exist a 'dynamic' memory network implementation in Keras. For an example of a 'dynamic' memory network see [this](https://github.com/ethancaballero/Improved-Dynamic-Memory-Networks-DMN-plus) repository. 

## Dependencies

* tensorflow (https://www.tensorflow.org/install/)
* functools
* tarfile
* re

## Usage

Run `python memorynetwork.py` in terminal and the code will begin training.

## Credits

Credits for the code go to the creator of Keras, [fchollet](https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py) i've merely created a wrapper to get people started.
