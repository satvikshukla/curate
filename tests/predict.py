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

def main():
	system('clear')

	print('Hi, welcome to pyArt!')
	sleep(2)
	print('\nPlease make sure that the image you want to analyze is in the list given below \n')

	images = get_images()

	input('\nPress Enter to continue')
	test_img = input('Enter the image name you want to analyze (including the format as shown in the list')

	if not (test_img in images):
		print('Sorry, the image is not present in the database')

if __name__ == '__main__':
	main()