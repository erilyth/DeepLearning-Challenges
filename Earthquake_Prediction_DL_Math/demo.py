import numpy as np

# Import data

data_total = []

with open('database.csv', "r") as f:
    f.readline()
    for line in f:
        data_point = []
        values = line.split(',')
        cur_date = values[0].split('/')
        # Consider only the year and month since they are highly significant
        # Time can be ignored.
        if len(cur_date) < 3:
            continue
        cur_year = cur_date[2]
        cur_month = cur_date[1]
        print cur_month, cur_year
        data_point.append(float(cur_year))
        data_point.append(float(cur_month))
        # Latitude
        data_point.append(float(values[2]))
        # Longitude
        data_point.append(float(values[3]))
        # Magnitude
        data_point.append(float(values[8]))
        data_total.append(data_point)

# Normalize the data
data_total = np.asarray(data_total)
column_means = data_total.mean(axis=0)
column_variance = data_total.std(axis=0)
data_total -= column_means
data_total /= column_variance

# Shuffle the data so they can be split into test and train sets
np.random.shuffle(data_total)

# Split the data into 80%, 20% training and testing sets
data_train = data_total[:(len(data_total)*8.0)/10.0]
data_test = data_total[(len(data_total)*8.0)/10.0:len(data_total)-1]

data_train_x, data_train_y = data_train[:,:4], data_train[:,4]
data_test_x, data_test_y = data_test[:,:4], data_test[:,4]


"""
# Sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
"""