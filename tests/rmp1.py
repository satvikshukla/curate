import numpy as np
import pandas as pd
import os
import time
from resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.models import model_from_json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
		if tmp_style == 'Renaissance' or tmp_style == 'Baroque':
			print(tmp_style)
			tmp_img = load_img(base_path + 'train_1/' + i, target_size=(224, 224))
			tmp_img = img_to_array(tmp_img)
			tmp_img = np.expand_dims(tmp_img, axis=0)
			tmp_img = preprocess_input(tmp_img)
			x_train.append(tmp_img)

			if tmp_style not in styles:
				styles[tmp_style] = counter
				counter = counter + 1

			y_train.append(styles.get(tmp_style))

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

json_file = open('model_json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('resnet_model.h5')

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

t = time.time()
hist = loaded_model.fit(x_t, y_t, batch_size=32, epochs=5, verbose=1, validation_data=(x_v, y_v))
print('training time %s' % (t- time.time()))
(loss, acc) = loaded_model.evaluate(x_v, y_v, batch_size=10, verbose=1)

print('loss={:.4f}, accuracy: {:.4f}%'.format(loss,acc * 100))

text_file = open('results.txt', 'w')
text_file.write('acc %.4f' % acc)
text_file.close()

model_json = loaded_model.to_json()
with open('model_json', 'w') as json_file:
	json_file.write(model_json)

loaded_model.save_weights('loaded_model.h5')

print('saved')