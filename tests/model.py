import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

random_seed = 1

def code_it(str):
	sum = 0
	for char in str:
		sum = sum + str.index(char) * ord(char)

	return sum

def read_file():
	base_path = './../../data/'
	raw_data = pd.read_csv(base_path + 'all_data_info.csv', dtype=object)
	data = pd.DataFrame(raw_data)
	images = os.listdir(base_path + 'train_t')

	# until future
	# relevant_col = ['artist', 'date', 'style', 'new_filename']
	relevant_col = ['style', 'new_filename']

	new_data = data[relevant_col]

	# for future change
	# new_data = new_data.dropna(subset=['artist', 'style', 'date'], how='any')
	# df['Tenant'].replace('', np.nan, inplace=True)
	# new_data['style'] = new_data['style'].replace('', np.nan)
	new_data = new_data.dropna(subset=['style', 'new_filename'], how='any')
	new_data['style hash'] = new_data['style'].apply(code_it)

	new_data = new_data.loc[new_data['new_filename'].isin(images), ]
	x_train = []
	y_train = []
	# j = 0
	
	for i in images:
		# print(i)
		tmp_img = load_img(base_path + 'train_t/' + i, target_size=(224, 224))
		tmp_img = img_to_array(tmp_img)
		tmp_img = np.expand_dims(tmp_img, axis=0)
		tmp_img = preprocess_input(tmp_img, mode='tf')
		x_train.append(tmp_img.flatten())
		tmp_val = new_data.loc[new_data['new_filename'] == i]['style hash'].values
		# print(j, tmp_val[0])
		# j = j + 1
		# print((new_data.loc[new_data['new_filename'] == i]['style'].values), tmp_val[0])
		y_train.append(tmp_val[0])

	del raw_data
	del data
	del new_data

	# print(y_train)
	x_train_data = pd.DataFrame(x_train, dtype=float)
	y_train_data = pd.Series(y_train, dtype=int)
	# x_train_data.to_csv('./../data/x-train.csv', sep=',')
	# y_train_data.to_csv('./../data/y-train.csv', sep=',')

	# print('x type', type(x_train_data))
	# print(y_train_data.shape)
	# print('y type', type(y_train_data))
	# print(y_train_data)
	x_train_data = x_train_data.values.reshape(-1, 224, 224, 3)

	y_train_data = to_categorical(y_train_data)
	# print(y_train_data)

	return (x_train_data, y_train_data)

def make_model(x_train_data, y_train_data):

	x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_data, 
														test_size=0.1, random_state=random_seed)

	model = Sequential()

	model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', 
						activation='relu', input_shape=(224, 224, 3)))
	model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', 
						activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', 
						activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', 
						activation='relu'))
	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))

	optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	learn_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, 
							verbose=1, factor=0.5, min_lr=0.00001)

	epochs = 1
	batch_size = 86

	datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
									featurewise_std_normalization=False, samplewise_std_normalization=False,
									zca_whitening=False, rotation_range=10, zoom_range=0.1, 
									width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False,
									vertical_flip=False)

	datagen.fit(x_train)

	history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
									epochs=epochs, validation_data=(x_val, y_val), verbose=2,
									steps_per_epoch=x_train.shape[0], callbacks=[learn_rate_reduction])

def main():
	x_train_data, y_train_data = read_file()
	make_model(x_train_data, y_train_data)

if __name__ == '__main__':
	main()