import numpy as np
import pandas as pd
from os import system, listdir
from time import time
from sys import exit
from math import ceil
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.applications.imagenet_utils import preprocess_input
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# def acc_top5(y_true, y_pred):
# 	return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def get_movements():
	file = open('./../data/movements.txt')
	movements_ls = [val.strip() for val in file]

	return movements_ls

def get_data(movements_ls, flag=True):
	random_seed = 1
	base_path = './../../data/'
	raw_data = pd.read_csv(base_path + 'all_data_info.csv', dtype=object)
	data = pd.DataFrame(raw_data)
	images = listdir(base_path + 'train')

	# until future
	# relevant_col = ['artist', 'date', 'style', 'new_filename']
	relevant_col = ['style', 'new_filename']

	new_data = data[relevant_col]
	
	del raw_data
	del data

	new_data = new_data.loc[new_data['new_filename'].isin(images), ]
	x_train = []
	y_train = []
	styles = {}
	counter = 0

	if not flag:
		file = open('./../data/dict.txt')
		styles = eval(file.read())

	for i in images:
		if not ((new_data.loc[new_data['new_filename'] == i]['style']).empty):
			tmp_style = new_data.loc[new_data['new_filename'] == i]['style'].values[0]
			# if tmp_style in movements_ls:
			if tmp_style == 'Baroque' or tmp_style == 'Impressionism':
				tmp_img = load_img(base_path + 'train/' + i, target_size=(224, 224))
				tmp_img = img_to_array(tmp_img)
				tmp_img = np.expand_dims(tmp_img, axis=0)
				tmp_img = preprocess_input(tmp_img)
				x_train.append(tmp_img)

				if tmp_style not in styles:
					styles[tmp_style] = counter
					counter = counter + 1
				
				y_train.append(styles.get(tmp_style))

	print(styles)

	# file = open('./../data/dict.txt', 'w')
	# file.write(str(styles))
	# file.close()

	del new_data

	x_train_data = np.array(x_train)
	x_train_data = np.rollaxis(x_train_data, 1, 0)
	x_train_data = x_train_data[0]

	num_classes = counter
	y_train_data = np_utils.to_categorical(y_train, num_classes)

	del x_train
	del y_train

	# x, y = shuffle(x_train_data, y_train_data, random_state=random_seed)
	# x = x / 255
	# x = x - np.mean(x, axis=0)

	# del x_train_data
	# del y_train_data

	x_t, x_v, y_t, y_v = train_test_split(x_train_data, y_train_data, test_size= 0.1, random_state=random_seed)

	del x_train_data
	del y_train_data

	# del x
	# del y

	return (x_t, x_v, y_t, y_v, num_classes)

def get_chunk(x_t, y_t):
	out = []
	length = ceil(len(x_t[:,0,0,0]) / 1000)
	# print(x_t.shape, y_t.shape)

	for i in range(length):
		out.append((x_t[i:1000 * (i + 1), :, :, :], y_t[i:1000 * (i + 1), :]))

	return out

def train_model(x_t, x_v, y_t, y_v, num_classes):
	img_input = Input(shape=(224, 224, 3))

	# model = ResNet50(input_tensor=img_input, include_top=True, weights=None, classes=num_classes)
	model = ResNet50(input_tensor=img_input, include_top=True)
	# model.summary()
	last_layer = model.get_layer('avg_pool').output
	x = Flatten(name='flatten')(last_layer)
	out = Dense(num_classes, activation='softmax', name='output_layer')(x)
	resnet_model = Model(inputs=img_input, outputs=out)
	# resnet_model.summary()

	for layer in resnet_model.layers[:-1]:
		layer.trainable = False

	resnet_model.layers[-1].trainable

	# resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.top_k_categorical_accuracy, acc_top5])
	resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

	datagen.fit(x_t)
	# i = 0

	t = time()
	resnet_model.fit_generator(datagen.flow(x_t, y_t, batch_size=16), steps_per_epoch=10, epochs=100)
	# for e in range(3):
	# 	print('epoch', e)
	# 	i = i + 1
	# 	j = 0
	# 	for X_train, Y_train in get_chunk(x_t, y_t): # these are chunks of ~10k pictures
	# 		print(X_train.shape, Y_train.shape)
	# 		j = j + 1
	# 		k = 1
	# 		for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=1000): # these are chunks of 32 samples
	# 			print('i =', i,  'j =', j, 'k =', k)
	# 			k = k + 1
	# 			loss = resnet_model.fit(X_batch, Y_batch, epochs=2, verbose=1)

	# resnet_model.fit(x_t, y_t, batch_size=32, epochs=100, verbose=1, validation_data=(x_v, y_v))
	print('training time %s' % (t- time()))
	(loss, acc) = resnet_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

	print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

	text_file = open('./../data/results_new.txt', 'w')
	text_file.write('acc %.4f' % acc)
	text_file.close()

	resnet_model.save('./../data/resnet_model_new.h5')

	print('saved')

def load_and_train(x_t, x_v, y_t, y_v):
	resnet_model = load_model('./../data/resnet_model.h5')

	t = time()
	resnet_model.fit(x_t, y_t, batch_size=32, epochs=100, verbose=1, validation_data=(x_v, y_v))
	print('training time %s' % (t- time()))
	(loss, acc) = resnet_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

	print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

	# text_file = open('./../data/results.txt', 'w')
	# text_file.write('acc %.4f' % acc)
	# text_file.close()

	# resnet_model.save('./../data/resnet_model.h5')

	print('saved')

def main():
	system('clear')

	movements_ls = get_movements()
	x = int(input('Enter 1 to train new model, 2 to train existing model, 3 to exit \n'))

	if x == 1:
		x_t, x_v, y_t, y_v, num_classes = get_data(movements_ls)
		train_model(x_t, x_v, y_t, y_v, num_classes)
	elif x == 2:
		x_t, x_v, y_t, y_v, num_classes = get_data(movements_ls, False)
		load_and_train(x_t, x_v, y_t, y_v)
	elif x == 3:
		exit()
	else:
		main()

if __name__ == '__main__':
	main()