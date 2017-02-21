# A classifier for cats and dogs

This is a response to Siraj's challenge of the week.

This is a classifier made with basic TensorFlow, using Transfer Learning from the Inception-V3 model. Data is taken from [this](https://www.kaggle.com/c/dogs-vs-cats) Kaggle competition, as recommended by Siraj. 

## Pipeline
JPG images --> Inception-V3 --> 2048-dimensional vector --> Fully connected layer --> Prediction

## How to use

### Training
First, train the model, run:

`python classifier.py train`

This script will call the getvector.py script, which uses the Inception-V3 model to produce 2048-dimensional vector representations of images. These 2048-dimensional vectors have already been saved in the data_inputs.txt and data_labels.txt files, for convenience. (It takes quite long on a CPU to run a few hundred images through the Inception network, let alone the full 25,000 image dataset.)

After getting these vector representations, I use a fully connected one layer neural network to output a prediction vector, built with TensorFlow.

For my training, I only used 300 images of cats and dogs combined. Hence the text files only contain 300 training example data. However, the network performs remarkably well for such few training examples. This is perhaps because the Inception model was trained on a lot of pictures of animals like cats and dogs so it's able to extract the relevant features.

### Testing
To test the model, run:

`python classifier.py test cat.jpg`

This runs the cat.jpg image through the Inception-V3 network to get the 2048-dimensional vector. Then it loads the saved TensorFlow one-layer neural network, and feeds the cat.jpg image vector into it. Do try other pictures too, the predictions are quite accurate.


## Notes:

I got a lot of the code from the TF Slim tutorial. It's taken from the [github](https://github.com/tensorflow/models/tree/master/slim). Specifically, I modified Inception-V3.py from [here](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v3.py) to return the 2048-D vector when the inception_v3 function is called, instead of the 1001 logits vector. The 1001 logits vector directly correspond to the 1001 categories that Inception is trained to classify.


P.S. made a grammatical error that I realised too late to correct - my variables should be named 'input' instead of 'inputs'
