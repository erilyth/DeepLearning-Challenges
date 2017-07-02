# Cats vs Dogs Classification

Given input images of cats and dogs, predict which one of them it is. Essentially a binary image classification task. Response for Sirajology's challenge [here](https://www.youtube.com/watch?v=cAICT4Al5Ow)

Dataset can be downloaded from [here](https://www.kaggle.com/c/dogs-vs-cats)

## Approach

* Read images as batches.
* Run Inception-V4 to extract 'Mixed_6a' layer features for transfer learning. The shape of the features is (1,17,17,1024).
* Add a few more convolution + maxpool layers on top of it.
* Finally add a fully connected layer with dropout to get the final classification results with 2 outputs.
* Used the Adam optimizer to improve training speed.

## Details

* In transfer learning we extract features from one of the layers of the network and then build our own network that we train on top of it. This greatly reduces the number of epochs and the amount of training data required to fit only a few final layers compared to the entire network.
* Specifically used 'Mixed_6a' layer features since a 17x17 image with 1024 filters seemed appropriate to build my own shallow Convolutional Net on top of it (compared to the initial image of a much larger size).
* Network shape: (Inception-V4 (till Mixed_6a), Conv2D, MaxPool, Conv2D, MaxPool, Fully Connected, Dropout) 
* Used 5000 images of cats and 5000 images of dogs from the dataset to achieve prediction accuracies of 80% with only a few hours of training. Using the entire dataset and training for longer would increase the accuracy further.

## Usage

Run the code with `python tfnet.py` and wait for it to complete a 1000 iterations.

## Credits

Sirajology for the starter code and Tensorflow [SLIM pretrained models](https://github.com/tensorflow/models/tree/master/slim#Pretrained)