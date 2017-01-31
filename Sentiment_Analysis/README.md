# Sentiment Analysis

Predict the sentiment in game titles from the IGN game dataset. Classification outputs a class from the set of 11 classes.
Classes: ('Great', 'Good', 'Okay', 'Mediocre', 'Amazing', 'Bad', 'Awful', 'Painful', 'Unbearable', 'Masterpiece')

Performing sentiment analysis on a small number of output classes such as 2 or 3 (Positive, Negative, Neutral) would give us much higher accuracies compared to using these 11 classes. The task of analyzing the sentiment at a very fine granular level is a hard task.

Results: Best accuracy with stacked GRUs and Dropout - 45.1%

## Dependencies

* pandas
* scikit-learn
* tflearn
* numpy

## Features

* Trained multiple architectures with various network shapes and dropout values
* Stacked [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory) and [GRUs](https://en.wikipedia.org/wiki/Gated_recurrent_unit) used to compare performance
* Variants of train/test split to compare performance

## Challenge

This is the code for Siraj's challenge on Sentiment Analysis [here](https://www.youtube.com/watch?v=si8zZHkufRY)

## Datasets

* Game sentiment prediction - IGN Dataset

## Results

|Layers 						|Dropouts			|Train/Test split	|Batch-size	|Results	|
| ----------------------------- | ----------------- | ----------------- | --------- | --------- |
|GRU(256), GRU(256)				|(0.9, 0.9)			|0.90				|32			|45.1%		|
|GRU(128), GRU(128)				|(0.95, 0.95)		|0.90				|32			|44.6%		|
|LSTM(128), LSTM(128)			|(0.8, 0.8)			|0.90				|32			|42.3%		|
|GRU(128)						|(0.9)				|0.90				|64			|41.5%		|
|LSTM(256), LSTM(256)			|(0.6, 0.6)			|0.95				|64			|41.3%		|
|LSTM(128)						|(0.9)				|0.85				|32			|41.2%		|
|LSTM(128), LSTM(128), LSTM(128)|(0.9, 0.9, 0.9)	|0.80				|32			|40.8%		|

## Usage

Run `python demo.py` in terminal and it should start training. Default epochs set to 20.
