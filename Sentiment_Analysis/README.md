# Sentiment Analysis

Predict the sentiment in game titles from the IGN game dataset. Classification outputs a class from the set of 11 classes.
Classes: ('Great', 'Good', 'Okay', 'Mediocre', 'Amazing', 'Bad', 'Awful', 'Painful', 'Unbearable', 'Masterpiece')

## Dependencies

* pandas
* scikit-learn
* tflearn
* numpy

## Features

* Trained multiple architectures with various network shapes and dropout values
* Stacked LSTMs and GRUs used to compare performance
* Variants of train/test split to compare performance

## Challenge

This is the code for Siraj's challenge on Sentiment Analysis [here](https://www.youtube.com/watch?v=si8zZHkufRY)

## Datasets

* Game sentiment prediction - IGN Dataset

## Results

|Layers 					|Dropouts			|Train/Test split	|Batch-size	|Results	|
| ------------------------- | ----------------- | ----------------- | --------- | --------- |
|LSTM(256), LSTM(256)		|(0.6, 0.6)			|0.95				|64			|41.09%		|
|LSTM(128), LSTM(128)		|(0.8, 0.8)			|0.90				|32			|42.30%		|
|LSTM(128)					|(0.9)				|0.85				|32			|			|

![Challenge Dataset](Analysis/challengedata.png?raw=true "Challenge Dataset")

## Usage

Run `python demo.py` in terminal and it should start training. Default epochs set to 20.
