import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

random_seed = 1

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
	new_data = new_data.dropna(subset=['style'], how='all')

	new_data = new_data.loc[new_data['new_filename'].isin(images), ]
	x_train = []
	y_train = []
	
	for i in images:
		tmp_img = load_img(base_path + 'train_t/' + i, target_size=(224, 224))
		tmp_img = img_to_array(tmp_img)
		tmp_img = np.expand_dims(tmp_img, axis=0)
		tmp_img = preprocess_input(tmp_img)
		x_train.append(tmp_img.flatten())
		tmp_val = new_data.loc[new_data['new_filename'] == i]['style'].values
		y_train.append(tmp_val[0])

	del raw_data
	del data
	del new_data

	x_train_data = pd.DataFrame(x_train, dtype=float)
	y_train_data = pd.Series(y_train, dtype='S')
	x_train_data.to_csv('./../data/x-train.csv', sep=',')
	y_train_data.to_csv('./../data/y-train.csv', sep=',')

	print('x type', type(x_train_data))
	print(y_train_data.shape)
	print('y type', type(y_train_data))
	x_train_data = x_train_data.values.reshape(-1, 224, 224, 3)

	# use hash to have nique integer values correspond to different categories
	# y_train_data = to_categorical(y_train_data)

def main():
	read_file()

if __name__ == '__main__':
	main()