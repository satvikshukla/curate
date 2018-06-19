import numpy as np
import pandas as pd
import os
import time
from sys import exit
from keras import metrics
from keras.models import Model, model_from_json, load_model
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# def acc_top5(y_true, y_pred):
# 	return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

def get_movements():
	file = open('./../data/movements.txt')
	movements_ls = [val.strip() for val in file]

	return movements_ls

def get_data(movements_ls):
	random_seed = 1
	base_path = './../../data/'
	raw_data = pd.read_csv(base_path + 'all_data_info.csv', dtype=object)
	data = pd.DataFrame(raw_data)
	images = os.listdir(base_path + 'train_1')

	# until future
	# relevant_col = ['artist', 'date', 'style', 'new_filename']
	relevant_col = ['style', 'new_filename']

	new_data = data[relevant_col]

	# for future change
	# new_data = new_data.dropna(subset=['artist', 'style', 'date'], how='any')
	new_data = new_data.dropna(subset=['style', 'new_filename'], how='any')

	new_data = new_data.loc[new_data['new_filename'].isin(images), ]
	x_train = []
	y_train = []
	styles = dict()
	counter = 0

	for i in images:
		if not ((new_data.loc[new_data['new_filename'] == i]['style']).empty):
			tmp_style = new_data.loc[new_data['new_filename'] == i]['style'].values[0]
			if tmp_style in movements_ls:
				tmp_img = load_img(base_path + 'train_1/' + i, target_size=(224, 224))
				tmp_img = img_to_array(tmp_img)
				tmp_img = np.expand_dims(tmp_img, axis=0)
				tmp_img = preprocess_input(tmp_img)
				x_train.append(tmp_img)

				if tmp_style not in styles:
					styles[tmp_style] = counter
					counter = counter + 1
				
				y_train.append(styles.get(tmp_style))

	print(styles)

	del raw_data
	del data
	del new_data

	x_train_data = np.array(x_train)
	x_train_data = np.rollaxis(x_train_data, 1, 0)
	x_train_data = x_train_data[0]
	# x_train_data.to_csv('./../data/x-train.csv', sep=',')
	# y_train_data.to_csv('./../data/y-train.csv', sep=',')

	num_classes = counter
	y_train_data = np_utils.to_categorical(y_train, num_classes)

	del x_train
	del y_train

	x, y = shuffle(x_train_data, y_train_data, random_state=random_seed)


	x_t, x_v, y_t, y_v = train_test_split(x_train_data, y_train_data, test_size= 0.1, random_state=random_seed)

	del x_train_data
	del y_train_data

	return (x_t, x_v, y_t, y_v, num_classes)

def train_model(x_t, x_v, y_t, y_v, num_classes):
	img_input = Input(shape=(224, 224, 3))

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

	t = time.time()
	hist = resnet_model.fit(x_t, y_t, batch_size=32, epochs=30, verbose=1, validation_data=(x_v, y_v))
	print('training time %s' % (t- time.time()))
	(loss, acc) = resnet_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

	print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

	text_file = open('./../data/results.txt', 'w')
	text_file.write('acc %.4f' % acc)
	text_file.close()

	resnet_model.save('./../data/resnet_model.h5')

	print('saved')

def load_and_train(x_t, x_v, y_t, y_v):
	resnet_model = load_model('./../data/resnet_model.h5')

	resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	t = time.time()
	hist = resnet_model.fit(x_t, y_t, batch_size=32, epochs=30, verbose=1, validation_data=(x_v, y_v))
	print('training time %s' % (t- time.time()))
	(loss, acc) = resnet_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

	print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

	text_file = open('./../data/results.txt', 'w')
	text_file.write('acc %.4f' % acc)
	text_file.close()

	resnet_model.save('./../data/resnet_model.h5')

	print('saved')

def main():
	os.system('clear')

	movements_ls = get_movements()
	x = int(input('Enter 1 to train new model, 2 to train existing model, 3 to exit \n'))

	x_t, x_v, y_t, y_v, num_classes = get_data(movements_ls)

	if x == 1:
		train_model(x_t, x_v, y_t, y_v, num_classes)
	elif x == 2:
		load_and_train(x_t, x_v, y_t, y_v)
	elif x == 3:
		exit()
	else:
		main()

if __name__ == '__main__':
	main()