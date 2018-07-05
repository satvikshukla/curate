from keras import backend as K

def content_reconstruction_loss(content, result):
	return K.sum(K.square(result - content))

def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

def style_reconstruction_loss(style, result, h, w):
	s = gram_matrix(style)
	r = gram_matrix(result)
	channels = 3
	return K.sum(K.square(s - r)) / float((2 * h * w * channels) ** 2)

def total_reconstruction_loss(x, h, w):
	val_one = K.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
	val_two = K.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
	return K.sum(K.pow(val_one + val_two, 1.25))