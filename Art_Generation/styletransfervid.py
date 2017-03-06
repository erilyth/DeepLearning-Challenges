import cv2

import numpy as np
import copy

import time
from PIL import Image

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

frames = []
vc = cv2.VideoCapture('temp.mp4')
c=1
if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False
while rval:
    c = c + 1
    if c%2 == 0:
    	frames.append(frame)
    rval, frame = vc.read()
vc.release()
print 'Frames collected:', len(frames)

height = 256
width = 256

# All the frames of the input video
content_imgs = []
for frame in frames:
	content_img = Image.fromarray(frame)
	content_img = content_img.resize((height, width))
	content_arr = np.asarray(content_img, dtype='float32')
	content_arr = np.expand_dims(content_arr, axis=0)
	content_arr[:, :, :, 0] -= 103.939
	content_arr[:, :, :, 1] -= 116.779
	content_arr[:, :, :, 2] -= 123.68
	# Convert from RGB to BGR
	content_arr = content_arr[:, :, :, ::-1]
	content_img = backend.variable(content_arr)
	content_imgs.append(content_img)

style_image_paths = ['images/styles/wave.jpg']
style_imgs = []
for style_img_path in style_image_paths:
	style_img = Image.open(style_img_path)
	style_img = style_img.resize((height, width))
	style_arr = np.asarray(style_img, dtype='float32')
	style_arr = np.expand_dims(style_arr, axis=0)
	style_arr[:, :, :, 0] -= 103.939
	style_arr[:, :, :, 1] -= 116.779
	style_arr[:, :, :, 2] -= 123.68
	# Convert from RGB to BGR
	style_arr = style_arr[:, :, :, ::-1]
	style_img = backend.variable(style_arr)
	style_imgs.append(style_img)

# Channels as the last dimension, using backend Tensorflow
combination_imgs = []
for t in range(len(frames)):
	comb_img = backend.placeholder((1, height, width, 3))
	combination_imgs.append(comb_img)

# We now finally have the content image variable, style image variables and combination image placeholder
# that we will concatenate to build a final input tensor to build our computation graph on top of, essentially
# we pass all these inputs to the network as if they are a part of a batch so they are all run parallely and we
# use the features generated to modify our combination image based on the losses

all_imgs = []
for content_img in content_imgs:
	all_imgs.append(content_img)
for style_img in style_imgs:
	all_imgs.append(style_img)
for comb_img in combination_imgs:
	all_imgs.append(comb_img)

input_tensor = backend.concatenate(all_imgs, axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])

content_weight = 0.025
style_weight = 5.0
tv_weight = 1.0

losses = []
for t in range(len(content_imgs)):
	loss = backend.variable(0.)
	losses.append(loss)

# --------------------------------------------------

def content_loss(content, combination):
	return backend.sum(backend.square(combination - content))

# Add content loss to this layer
layer_features = layers['block2_conv2']
for content_idx in range(len(content_imgs)):
	content_features = layer_features[content_idx, :, :, :]
	combination_features = layer_features[len(content_imgs) + len(style_imgs) + content_idx, :, :, :]
	losses[content_idx] += content_weight * content_loss(content_features, combination_features)

# --------------------------------------------------

def gram_matrix(x):
	features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
	gram = backend.dot(features, backend.transpose(features))
	return gram

def style_loss(style, combination):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = height * width
	return backend.sum(backend.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
for content_idx in range(len(content_imgs)):
	for layer_name in feature_layers:
		layer_features = layers[layer_name]
		for style_img_idx in range(len(style_imgs)):
			style_features = layer_features[len(content_imgs) + style_img_idx, :, :, :]
			combination_features = layer_features[len(content_imgs) + len(style_imgs) + content_idx, :, :, :]
			style_l = style_loss(style_features, combination_features)
			losses[content_idx] += (style_weight / (len(feature_layers)*len(style_imgs))) * style_l

# ---------------------------------------------------

# Total variation loss to ensure the image is smooth and continuous throughout

def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

for content_idx in range(len(content_imgs)):
	losses[content_idx] += tv_weight * total_variation_loss(combination_imgs[content_idx])

# ----------------------------------------------------

# Since all of these variables are nodes in our computational graph, we can directly
# calculate the gradients
grads = backend.gradients(losses, combination_imgs)

outputs = losses
outputs += grads
# Create the function from input combination_img to the loss and gradients
f_outputs = backend.function(combination_imgs, outputs)

# We finally have the gradients and losses at the combination_img computed as variables
# and we can use any standard optimization function to optimize combination_img

def eval_loss_and_grads(x):
	x = x.reshape((len(content_imgs), 1, height, width, 3))
	xs = []
	for el in x:
		el1 = el.reshape((1, height, width, 3))
		xs.append(el1)
	outs = f_outputs(xs)
	loss_value = 0.0
	for idx in range(len(content_imgs)):
		loss_value += outs[idx]
	grad_values = []
	for idx in range(len(content_imgs)):
		grad_values.append(outs[len(content_imgs)+idx])
	grad_values = np.asarray(grad_values)
	return loss_value, grad_values.flatten().astype('float64')

class Evaluator(object):
	def __init__(self):
		self.loss_value = None
		self.grads_values = None
	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value
	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

evaluator = Evaluator()

xs = []
for idx in range(len(content_imgs)):
	x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.0
	xs.append(x)

iters = 10

video = cv2.VideoWriter('video-out.avi',-1,1,(width,height))

for i in range(iters):
	print('Start of iteration', i)
	start_time = time.time()
	xs, min_val, info = fmin_l_bfgs_b(evaluator.loss, xs, fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	end_time = time.time()
	print('Iteration %d completed in %ds' % (i, end_time - start_time))

	x1 = copy.deepcopy(xs)
	x1 = x1.reshape((len(content_imgs), 1, height, width, 3))
	for idx in range(len(content_imgs)):
		x2 = x1[idx]
		x2 = x2.reshape((height, width, 3))
		# Convert back from BGR to RGB to display the image
		x2 = x2[:, :, ::-1]
		x2[:, :, 0] += 103.939
		x2[:, :, 1] += 116.779
		x2[:, :, 2] += 123.68
		x2 = np.clip(x2, 0, 255).astype('uint8')
		if i == iters - 1:
			video.write(x2)
			img_final = Image.fromarray(x2)
			img_final.save('result' + str(i) + str(idx) + '.bmp')

video.release()
