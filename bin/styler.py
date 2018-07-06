import time
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from utils import preprocess_image, deprocess_image
from loss import style_reconstruction_loss, content_reconstruction_loss, total_reconstruction_loss

class Styler(object):

	def __init__(self, base_img_path, style_img_path, output_img_path,
				content_weight, style_weight, tv_weight,
				content_layer, style_layers, iterations):

		self.base_img_path = base_img_path
		self.style_img_path = style_img_path
		self.output_img_path = output_img_path

		# w, h = load_img(self.base_img_path).size
		# dims = (h, w)

		width, height = load_img(base_image_path).size
		h = 400
		w = int(width * h / height)
		dims = (h, w)

		self.height = h
		self.width = w

		self.content_img = K.variable(preprocess_image(self.base_img_path, dims))
		self.style_img = K.variable(preprocess_image(self.style_img_path, dims))

		if K.image_dim_ordering() == 'th':
			self.output_img = K.placeholder((1, 3, dims[0], dims[1]))
		else:
			self.output_img = K.placeholder((1, dims[0], dims[1], 3))

		print("\tSize of content image is: {}".format(K.int_shape(self.content_img)))
		print("\tSize of style image is: {}".format(K.int_shape(self.style_img)))
		print("\tSize of output image is: {}".format(K.int_shape(self.output_img)))

		self.input_img = K.concatenate([self.content_img, self.style_img, self.output_img], axis=0)

		self.iterations = iterations

		self.content_weight = content_weight
		self.style_weight = style_weight
		self.tv_weight = tv_weight

		self.content_layer = content_layer
		self.style_layers = style_layers

		self.model = vgg19.VGG19(input_tensor=self.input_img, weights='imagenet', include_top=False)

		output_dict = dict([(layer.name, layer.output) for layer in self.model.layers])

		content_features = output_dict[self.content_layer]

		base_image_features = content_features[0, :, :, :]
		combination_features = content_features[2, :, :, :]

		content_loss = self.content_weight *  content_reconstruction_loss(base_image_features, combination_features)

		style_loss = K.variable(0.0)
		weight = 1.0 / float(len(self.style_layers))

		for layer in self.style_layers:
			style_features = output_dict[layer]
			style_image_features = style_features[1, :, :, :]
			output_style_features = style_features[2, :, :, :]
			style_loss = style_loss + weight * style_reconstruction_loss(style_image_features, output_style_features, self.height, self.width)

		style_loss = self.style_weight * style_loss

		total_variation_loss = self.tv_weight * total_reconstruction_loss(self.output_img, self.height, self.width)

		total_loss = content_loss + style_loss + total_variation_loss

		grads = K.gradients(total_loss, self.output_img)

		outputs = [total_loss]

		if type(grads) in {list, tuple}:
			outputs = outputs + grads
		else:
			outputs.append(grads)

		self.loss_and_grads = K.function([self.output_img], outputs)
	
	def style(self):
		print('styling...')
		if K.image_dim_ordering() == 'th':
			x = np.random.uniform(0, 255, (1, 3, self.height, self.width)) - 128
		else:
			x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128

		for i in range(self.iterations):
			x, _, _ = fmin_l_bfgs_b(self.loss, x.flatten(), fprime=self.grads, maxfun=20)

			img = deprocess_image(x.copy(), self.height, self.width)
			fname = self.output_img_path + 'at_itr_%d.png' % (i + 1)
			imsave(fname, img)

	def loss(self, x):
		if K.image_dim_ordering() == 'th':
			x = np.random.uniform(0, 255, (1, 3, self.height, self.width)) - 128
		else:
			x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128
		out = self.loss_and_grads([x])
		loss_val = out[0]
		return loss_val

	def grads(self, x):
		if K.image_dim_ordering() == 'th':
			x = np.random.uniform(0, 255, (1, 3, self.height, self.width)) - 128
		else:
			x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128
		out = self.loss_and_grads([x])

		if len(out[1:]) == 1:
			grad_values = out[1].flatten().astype('float64')
		else:
			grad_values = np.array(out[1:]).flatten().astype('float64')

		return grad_values

	def get_x(self):
		if K.image_dim_ordering() == 'th':
			x = np.random.uniform(0, 255, (1, 3, self.height, self.width)) - 128
		else:
			x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128
		
		return x