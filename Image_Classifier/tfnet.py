import tensorflow as tf
import numpy as np
import os
import inception_preprocessing
import inception_v4

from inceptionfeatures import getfeatures

# Parameters
learning_rate = 0.001
training_iters = 10
display_step = 1

# Network Parameters
dropout = 0.75 # Dropout 25%

# tf Graph input
x = tf.placeholder(tf.float32, [None, 17, 17, 1024])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def getfeatures(file_name):
    # Extract the features from InceptionNet (The features of Mixed_6a layer)
    slim = tf.contrib.slim
    image_size = inception_v4.inception_v4.default_image_size
    checkpoints_dir = os.getcwd()
    with tf.Graph().as_default():
        image_path = tf.read_file(file_name)
        image = tf.image.decode_jpeg(image_path, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            vector = inception_v4.inception_v4(processed_images, num_classes=1001, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v4.ckpt'), slim.get_model_variables('InceptionV4'))
        with tf.Session() as sess:
            init_fn(sess)
            np_image, vector = sess.run([image, vector])
        vector = np.asarray(vector)
        # print vector[1]['Mixed_6a'].shape
        return vector[1]['Mixed_6a']

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 17, 17, 1024])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1024, 1024])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 1024, 1024])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*1024, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 2]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([1024])),
    'bc2': tf.Variable(tf.random_normal([1024])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([2]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

avg_acc = 0.0
acc_check_points = 64 # Reset every 64 images
acc_cur = 1

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        # Consider the first 1000 images
        for file_id in range(0, 1000):
            acc_cur += 1
            catpath = 'train/cat.' + str(file_id) + '.jpg'
            dogpath = 'train/dog.' + str(file_id) + '.jpg'
            inp_x1, inp_y1 = getfeatures(catpath)[0], [0,1]
            inp_x2, inp_y2 = getfeatures(dogpath)[0], [1,0]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: [inp_x1, inp_x2], y: [inp_y1, inp_y2],
                                           keep_prob: dropout})
            if acc_cur % acc_check_points == 0:
                acc_cur = 1
                print "Accuracy:", avg_acc * 1.0 / acc_check_points
                avg_acc = 0.0

            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: [inp_x1, inp_x2],
                                                              y: [inp_y1, inp_y2],
                                                              keep_prob: 1.})
            print "Current Prediction: ", loss, acc
            avg_acc += acc
        step += 1
    print("Optimization Finished!")
