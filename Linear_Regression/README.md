# Linear Regression

Given data with n parameters (where the last parameter is our output), predict a linear regression line between each of the n-1 features wrt the output parameter.

## Dependencies

* pandas
* scikit-learn
* matplotlib
* numpy

You can just run
`pip install -r requirements.txt` 
in terminal to install the necessary dependencies. Here is a link to [pip](https://pip.pypa.io/en/stable/installing/) if you don't already have it.

## Features

* Since different datasets have different delimiters, number of features etc, first parse the dataset and create a clean version with a fixed delimiter and store as 'temporary.txt' and use that later on.

## Challenge

This is the code for Siraj's challenge on Linear Regression [here](https://www.youtube.com/watch?v=vOppzHpvTiQ&feature=youtu.be)

## Datasets

* Diabetes - (Age, Deficit, C_peptide) - Given the Age and Deficit, predict the amount of C_peptide
* Electrical Length - (Inhabitants, Distance, Length) - Given the number of inhabitants and distance, predict the length required

## Results

* Challenge Dataset

![Challenge Dataset](Analysis/challengedata.png?raw=true "Challenge Dataset")

* Diabetes Dataset

![Diabetes Dataset 1](Analysis/diabetes1.png?raw=true "Diabetes Dataset - Age vs C_peptide")
![Diabetes Dataset 2](Analysis/diabetes2.png?raw=true "Diabetes Dataset - Deficit vs C_peptide")

* Electrical Length Dataset

![Electrical Length Dataset 1](Analysis/ele1.png?raw=true "Electrical Length Dataset - Inhabitants vs Length")
![Electrical Length Dataset 2](Analysis/ele2.png?raw=true "Electrical Length Dataset - Distance vs Length")

## Usage

Type `python demo.py <datafile_location>` into terminal and you'll see the scatter plots and lines of best fit appear for the (feature, output) pairs for each feature.
