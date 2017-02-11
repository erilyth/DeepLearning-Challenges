# Train a neural network to predict Earthquake Magnitudes

This is the code for "How to Do Math Easily - Intro to Deep Learning #4's challenge by Siraj Raval on youtube.

## Overview

This is the code for [this](https://youtu.be/N4gDikiec8E) video on Youtube by Siraj Raval apart of the 'Intro to Deep Learning' Udacity nanodegree course. We build a 3 layer feedforward neural network trains on a set of binary number input data and predict the binary number output.

## Dependencies

* Numpy

## Usage

Run the demo.py script by running `python demo.py` in terminal.

## Features

* Used the earthquake magnitude prediction dataset.
* Added learning rate along with decay based on performance so that the learning curve can be controlled well.
* Split data into training batches which helped ease the computation to avoid dealing with very large matrices.
* Varied hyperparameters such as learning rate, number of hidden layers to measure and compare how the errors vary.
* Iterations per test varied based on the number of hidden units since larger networks take longer to train.

## Results

|Learning Rate 	|Hidden Units	|Error	|
| ------------- | ------------- | ----- |
|0.1			|6				|0.8054	|
|0.1			|12				|0.8090	|
|0.1			|24				|0.8224	|
|0.2			|6				|0.8056	|
|0.2			|12				|0.8089	|
|0.2			|24				|0.8005	|
|0.5			|6				|0.8035	|
|0.5			|12				|0.8086	|
|0.5			|24				|0.8126	|


##Credits

Credits for the original code go to [Andrew Trask](http://iamtrask.github.io/2015/07/12/basic-python-network/)

