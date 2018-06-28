import numpy as np
import pandas as pd
from os import system, listdir
from time import time
from sys import exit
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

def get_movements():
	file = open('./../data/movements.txt')
	movements_ls = [val.strip() for val in file]

	return movements_ls

def get_images():
	base_path = './../images'
	images = listdir(base_path)

	print('\n '.join(images))
	return(images)

def get_data():
	base_path = './../../data/'
	raw_data = pd.read_csv(base_path + 'all_data_info.csv', dtype=object)
	data = pd.DataFrame(raw_data)
	relevant_col = ['artist', 'style', 'new_filename']
	new_data = data[relevant_col]

	return new_data

def get_art_movement(image_path):
	model = load_model('./../data/resnet_model.h5')

	try:
		img = load_img('./../images/' + image_path, target_size=(224, 224))
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		preds = model.predict(img)
		styles = {}

		file = open('./../data/dict.txt')

		styles = eval(file.read())

		index = np.argmax(preds)

		for key, val in styles.items():
			if val == index:
				print(key, 'with precentage prediction of', np.max(preds))
		
		return key
	except:
		print('\nOh! An unexpected error occured in using input file. Please try different file.')
		return 0

def get_artist(image_path, art_movement, movements_ls, data):
	new_data = data.loc[data['style'] == art_movement]
	possible_artists = new_data['artist'].tolist()
	possible_artists = list(set(possible_artists))

	model = load_model('./../data/some_model.h5')

	try:
		img = load_img('./../images/' + image_path, target_size=(224, 224))
		img = img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)

		preds = model.predict(img)
		preds = preds.tolist()
		tmp_ls = []
		artists = {}

		file = open('./../data/some_dict.txt')
		artists = eval(file.read())

		for i, val in enumerate(preds):
			tmp_ls.append((i, val))

		tmp_ls.sort(key=lambda tup: tup[1], reverse=True)

		for i, val_one in tmp_ls:
			for j, val_two in artists:
				if i == val_two and flag:
					print('most probable artist is', j)
					return 1
		
		return 0
	except:
		print('\nOh! An unexpected error occured in using input file. Please try different file.')
		return 0

def go_on():
	start_over = input('\nEnter yes if you want to test more images: ')

	if start_over.lower() == 'yes':
		main()
	else:
		exit()

def main():
	system('clear')

	print('Hi, welcome to pyArt!')
	print('\nPlease make sure that the image you want to analyze is in the list given below')

	images = get_images()
	movements_ls = get_movements()

	test_img = input('\nEnter the image name you want to analyze (including the format as shown in the list): ')

	if not (test_img in images):
		print('\nSorry, the image is not present in the database')
		go_on()
	else:
		image_path = test_img
		returned_movement = get_art_movement(image_path)

		if returned_movement == 0:
			go_on()

		data = get_data()
		returned_artist = get_artist(image_path, returned_movement, movements_ls, data)
		go_on()

if __name__ == '__main__':
	main()