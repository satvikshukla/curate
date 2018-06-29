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

def get_list(name):
	string_to_get = './../data/' + name + '.txt'
	
	file_to_get = open(string_to_get)
	ls = [val.strip() for val in file_to_get]

	return ls

def get_data(name, ls, flag=True):
	if name == 'movements':
		relevant_col = ['style', 'new_filename']
		match_string = 'style'
	else:
		relevant_col = ['artist', 'new_filename']
		match_string = 'artist'

	base_path = './../../data/'
	raw_data = pd.read_csv(base_path + 'all_data_info.csv', dtype=object)
	data = pd.DataFrame(raw_data)
	images = listdir(base_path + 'train_o')

	new_data = data[relevant_col]
	
	del raw_data
	del data

	new_data = new_data.loc[new_data['new_filename'].isin(images), ]
	x_train = []
	y_train = []
	ref_dict = {}
	counter = 0

	if not flag:
		string_to_get = './../data/dict_' + name + '.txt'
		file_to_get = open(string_to_get)
		ref_dict = eval(file_to_get.read())

	for i in images:
		if not ((new_data.loc[new_data['new_filename'] == i][match_string]).empty):
			tmp_string = new_data.loc[new_data['new_filename'] == i][match_string].values[0]
			if tmp_string in ls:
				tmp_img = load_img(base_path + 'train_o/' + i, target_size=(224, 224))
				tmp_img = img_to_array(tmp_img)
				tmp_img = np.expand_dims(tmp_img, axis=0)
				tmp_img = preprocess_input(tmp_img)
				x_train.append(tmp_img)

				if tmp_string not in ref_dict:
					ref_dict[tmp_string] = counter
					counter = counter + 1
				
				y_train.append(ref_dict.get(tmp_string))

	print(ref_dict)

	string_to_write = './../data/dict_' + name + '.txt'
	file_to_write = open(string_to_write, 'w')
	file_to_write.write(str(ref_dict))
	file_to_write.close()

	del new_data

	x_train_data = np.array(x_train)
	x_train_data = np.rollaxis(x_train_data, 1, 0)
	x_train_data = x_train_data[0]

	num_classes = counter
	y_train_data = np_utils.to_categorical(y_train, num_classes)

	del x_train
	del y_train

	# x, y = shuffle(x_train_data, y_train_data, random_state=random_seed)

	# del x_train_data
	# del y_train_data

	x_t, x_v, y_t, y_v = train_test_split(x_train_data, y_train_data, test_size= 0.1)

	del x_train_data
	del y_train_data

	# del x
	# del y

	return (x_t, x_v, y_t, y_v, num_classes)

def train_model(x_t, x_v, y_t, y_v, num_classes):
	img_input = Input(shape=(224, 224, 3))

	model = ResNet50(input_tensor=img_input, include_top=True)
	last_layer = model.get_layer('avg_pool').output
	x = Flatten(name='flatten')(last_layer)
	out = Dense(num_classes, activation='softmax', name='output_layer')(x)
	resnet_model = Model(inputs=img_input, outputs=out)

	for layer in resnet_model.layers[:-1]:
		layer.trainable = False

	resnet_model.layers[-1].trainable

	# resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.top_k_categorical_accuracy, acc_top5])
	resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

	datagen.fit(x_t)

	t = time()
	resnet_model.fit_generator(datagen.flow(x_t, y_t, batch_size=32), steps_per_epoch=5, epochs=10)
	print('training time %s' % (t- time()))
	(loss, acc) = resnet_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

	print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

	text_file = open('./../data/results_movements_tmp.txt', 'w')
	text_file.write('acc %.4f' % acc)
	text_file.close()

	resnet_model.save('./../data/resnet_model_movements_tmp.h5')

	print('saved')

def load_and_train(x_t, x_v, y_t, y_v):
	resnet_model = load_model('./../data/resnet_model_artists_tmp.h5')

	datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

	datagen.fit(x_t)

	t = time()
	resnet_model.fit_generator(datagen.flow(x_t, y_t, batch_size=32), steps_per_epoch=5, epochs=10)
	print('training time %s' % (t- time()))
	(loss, acc) = resnet_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

	print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

	text_file = open('./../data/results_artists_tmp_two.txt', 'w')
	text_file.write('acc %.4f' % acc)
	text_file.close()

	resnet_model.save('./../data/resnet_model_artists_tmp_two.h5')

	print('saved')

def main():
	system('clear')

	movements_ls = get_list('movements')
	artists_ls = get_list('artists')
	x = int(input('Enter 1 to train new movements, 2 to train existing movements, 3 train new artists, 4 to train existing artists: '))

	if x == 1:
		x_t, x_v, y_t, y_v, num_classes = get_data('movements', movements_ls)
		train_model(x_t, x_v, y_t, y_v, num_classes)
	elif x == 2:
		x_t, x_v, y_t, y_v, num_classes = get_data('movements', movements_ls, False)
		load_and_train(x_t, x_v, y_t, y_v)
	elif x == 3:
		x_t, x_v, y_t, y_v, num_classes = get_data('artists', artists_ls)
		train_model(x_t, x_v, y_t, y_v, num_classes)
	elif x == 4:
		x_t, x_v, y_t, y_v, num_classes = get_data('artists', artists_ls, False)
		load_and_train(x_t, x_v, y_t, y_v)
	else:
		main()

if __name__ == '__main__':
	main()