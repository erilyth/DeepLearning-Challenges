import pandas as pd
from sklearn import linear_model

import matplotlib.pyplot as plt

import numpy as np
import sys
import re


data_file = sys.argv[1]
dimensions = 2 # Default

"""
Parse the given data and try convert it into a standard format with comma (",") as the delimiter
and save it temporarily in 'temporary.txt'. This modified data can also be used with ease elsewhere.
"""
data = open(data_file, 'r')
data = data.read().splitlines()
new_data = open('temporary.txt', 'w')
for line in data:
	data_cur = filter(None, re.split("[, \-:;]+", line))
	dimensions = len(data_cur)
	new_d = ""
	for d in range(len(data_cur)):
		new_d += str(data_cur[d])
		if d != len(data_cur) - 1:
			new_d += "," # Use comma as the standard delimiter
	new_d += "\n"
	new_data.write(new_d)
new_data.close()

"""
Extract data using pandas and store them for later use
"""
col_names = []
for i in range(dimensions):
	col_names.append(str(i))
#read data
dataframe = pd.read_table(open('temporary.txt'), sep=',', header=None, names=col_names, lineterminator='\n')
x_data = []
for col_name in col_names[:]:
	# Get each column
	x_values = dataframe[[col_name]].as_matrix().T[0]
	x_data.append(x_values)
x_data = np.asarray(x_data)
x_final = x_data.T
y_final = dataframe[[col_names[len(col_names)-1]]].as_matrix().T[0]

"""
Regress each variable wrt the output variable that we want to predict. 
If data is of n dimensions, we get n-1 plots for each variable with the output variable that we are trying to predict.
"""
for dim in range(dimensions-1):
	#train model on data
	x = x_final.T[dim]
	linear_reg = linear_model.LinearRegression()
	linear_reg.fit(x.reshape(-1,1), y_final)
	# The score of the fit that tells us how well the linear regression line fits the data
	print linear_reg.score(x.reshape(-1,1), y_final)
	#visualize results
	plt.scatter(x, y_final)
	plt.plot(x, linear_reg.predict(x.reshape(-1,1)))
	plt.show()
