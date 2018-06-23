import numpy as np
from os import system, listdir
from time import time, sleep
from sys import exit
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input


def get_images():
	base_path = './../images'
	images = listdir(base_path)

	print('\n '.join(images))
	return(images)

def process(image_path):
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
	except:
		print('\nOh! An unexpected error occured in using input file. Please try different file.')
		sleep(1)
		go_on()

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

	test_img = input('\nEnter the image name you want to analyze (including the format as shown in the list): ')

	if not (test_img in images):
		print('\nSorry, the image is not present in the database')
		go_on()
	else:
		image_path = test_img
		process(image_path)
		go_on()

if __name__ == '__main__':
	main()