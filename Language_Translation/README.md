# How_to_make_a_language_translator
This is the code for "How to Make a Language Translator - Intro to Deep Learning #11' by Siraj Raval on YouTube

# Coding Challenge - Due Date Thursday March 30th at 12 PM PST

Make your own language translator using a deep learning architecture of your choice in Tensorflow. It doesn't have to use Neural Machine Translation specifically. [Here](https://github.com/search?utf8=%E2%9C%93&q=language+translation+language%3A%22Jupyter+Notebook%22+&type=) are some example. This exercise will help you appreciate the full power of statistical translation techniques.

## Overview

This is the code for [this](https://youtu.be/nRBnh4qbPHI) video on Youtube by Siraj Raval as part of the Deep Learning Nanodegree course with Udacity. This code implemnents [Neural Machine Translation](https://github.com/neubig/nmt-tips) which is what Google Translate uses to translate Languages.

## Dependencies

* tensorflow
* nltk 
* six

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/)


## Usage

To train model on data and test it to compute the [BLEU score](https://en.wikipedia.org/wiki/BLEU) run this:

``python translate.py source_language target_language`` (i.e. python translate.py fr en for fr->en translation)

Training results are shown in plot_training_process.ipynb.

Testing results and findings are discussed in the paper.

## Credits

The credits for this code go to [fanshi118](https://github.com/fanshi118/NLP_NMT_Project). I've merely created a wrapper to get people started.
