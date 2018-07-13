import numpy as np
import time
from argparse import ArgumentParser
from keras import backend as K
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b

def main():

	desc = 'style transfer keras'
	parser = ArgumentParser(description=desc)
	parser.add_argument('base_img_path', metavar='base', type=str,
						help='path to base image.')
	parser.add_argument('style_img_path', metavar='style', type=str,
						help='path to style image.')
	parser.add_argument('output_img_path', metavar='output', type=str,
						help='path to output directory.')
	parser.add_argument('--iter', type=int, default=10,
						metavar='iterations', help='Number of iterations.')
	parser.add_argument('--content_weight', type=float, default=0.025,
						help='Weight for content feature loss')
	parser.add_argument('--style_weight', type=float, default=1.0,
						help='Weight for style feature loss')
	parser.add_argument('--tv_weight', type=float, default=1.0,
						help='Weight for total variation loss')

	args = parser.parse_args()
	base_image_path = args.base_img_path
	style_image_path = args.style_img_path
	output_image_path = args.output_img_path
	iterations = args.iter

	total_variation_weight = args.tv_weight
	style_weight = args.style_weight
	content_weight = args.content_weight

	w, h = load_img(base_image_path).size
	height = 400
	width = int(w * height / h)
	dimension = (height, width)

	def preprocess_image(image_path):
		img = load_img(image_path, target_size=dimension)
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = vgg19.preprocess_input(img)
		return img

	def deprocess_image(x):
		if K.image_data_format() == 'channels_first':
			x = x.reshape((3, height, width))
			x = x.transpose((1, 2, 0))
		else:
			x = x.reshape((height, width, 3))

		x[:, :, 0] = x[:, :, 0] + 103.939
		x[:, :, 1] = x[:, :, 1] + 116.779
		x[:, :, 2] = x[:, :, 2] + 123.68

		x = x[:, :, ::-1]
		x = np.clip(x, 0, 255).astype('uint8')
		return x

	base_image = K.variable(preprocess_image(base_image_path))
	style_image = K.variable(preprocess_image(style_image_path))

	if K.image_data_format() == 'channels_first':
		combination_image = K.placeholder((1, 3, height, width))
	else:
		combination_image = K.placeholder((1, height, width, 3))

	input_tensor = K.concatenate([base_image, style_image, combination_image], axis=0)

	model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

	output_dict = dict([(layer.name, layer.output) for layer in model.layers])

	def gram_matrix(x):
		if K.image_data_format() == 'channels_first':
			features = K.batch_flatten(x)
		else:
			features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
		gram = K.dot(features, K.transpose(features))
		return gram

	def content_loss(base, result):
		return K.sum(K.square(result - base))

	def style_loss(style, combination):
		S = gram_matrix(style)
		C = gram_matrix(combination)
		channels = 3
		size = height * width
		return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

	def total_variation_loss(x):
		if K.image_data_format() == 'channels_first':
			a = K.square(x[:, :, :height - 1, :width - 1] - x[:, :, 1:, :width - 1])
			b = K.square(x[:, :, :height - 1, :width - 1] - x[:, :, :height - 1, 1:])
		else:
			a = K.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
			b = K.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
		return K.sum(K.pow(a + b, 1.25))

	loss = K.variable(0.)
	content_layer = output_dict['block5_conv2']
	base_image_feaures = content_layer[0, :, :, :]
	combination_image_features = content_layer[2, :, :, :]
	loss = loss + content_weight * content_loss(base_image_feaures, combination_image_features)

	style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

	for layer_name in style_layers:
		layer_features = output_dict[layer_name]
		style_image_features = layer_features[1, :, :, :]
		combination_image_features = layer_features[2, :, :, :]
		tmp_val = style_loss(style_image_features, combination_image_features)
		loss = loss + (style_weight / len(style_layers)) * tmp_val
	loss = loss + total_variation_weight * total_variation_loss(combination_image)

	grads = K.gradients(loss, combination_image)

	outputs = [loss]
	if isinstance(grads, (list, tuple)):
		outputs = outputs + grads
	else:
		outputs.append(grads)

	f_outputs = K.function([combination_image], outputs)

	def loss_and_grads(x):
		if K.image_data_format() == 'channels_first':
			x = x.reshape((1, 3, height, width))
		else:
			x = x.reshape((1, height, width, 3))
		outs = f_outputs([x])
		loss_value = outs[0]

		if len(outs[1:]) == 1:
			grad_values = outs[1].flatten().astype('float64')
		else:
			grad_values = np.array(outs[1:]).flatten().astype('float64')
		return loss_value, grad_values

	class Styler(object):

		def __init__(self):
			self.loss_value = None
			self.grad_values = None

		def loss(self, x):
			loss_value, grad_values = loss_and_grads(x)
			self.loss_value = loss_value
			self.grad_values = grad_values
			return self.loss_value

		def grads(self, x):
			grad_values = np.copy(self.grad_values)
			self.grad_values = None
			self.loss_value = None
			return grad_values

	styler = Styler()

	x = preprocess_image(base_image_path)

	for i in range(iterations):
		print('starting', i)
		x, _, _ = fmin_l_bfgs_b(styler.loss, x.flatten(), fprime=styler.grads, maxfun=20)
		img = deprocess_image(x.copy())
		fname = output_image_path + 'at_itr_%d.png' % i
		save_img(fname, img)
		print('saved as', fname)

if __name__ == '__main__':
	main()