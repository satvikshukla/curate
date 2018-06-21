import numpy as np
import pandas as pd
from os import system, listdir
from time import time, sleep
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

def get_images():
	base_path = './../images'
	images = listdir(base_path)

	print('\n '.join(images))
	return(images)

def process(image_path):
	model = load_model('./../data/resnet_model_three_cat.h5')
	# model.summary()
	# model = ResNet50(weights='imagenet')
	# print(type(model), type(model_t))
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
			print(key)
	
	# print(preds)
	# print(type(preds))
	# print(type(preds[0]))
	# print(type(preds[0][0]))
	# print(np.argmax(preds))

	# print(decode_predictions(preds, top=1)[0])

def main():
	system('clear')

	print('Hi, welcome to pyArt!')
	sleep(2)
	print('\nPlease make sure that the image you want to analyze is in the list given below')

	images = get_images()

	input('\nPress Enter to continue')
	test_img = input('Enter the image name you want to analyze (including the format as shown in the list: ')

	if not (test_img in images):
		print('Sorry, the image is not present in the database')
	else:
		image_path = test_img
		process(image_path)

if __name__ == '__main__':
	main()