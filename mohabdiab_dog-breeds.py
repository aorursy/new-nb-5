# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
main_dir = "../input/"

train_dir = "dog-breed-identification/train"

path = os.path.join(main_dir,train_dir)



for p in os.listdir(path):

    category = p.split(".")[0]

    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

    new_img_array = cv2.resize(img_array, dsize=(80, 80))

    plt.imshow(new_img_array,cmap="gray")

    break


from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation, Flatten, Dropout, BatchNormalization



import h5py

import keras as k
num_classes = 120

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



model = Sequential()

model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

model.layers[0].trainable = False
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
import os

from tqdm import tqdm

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/dog-breed-identification/labels.csv')

df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)
im_size = 90
x_train = []

y_train = []

x_test = []
i = 0 

for f, breed in tqdm(df_train.values):

    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))

    label = one_hot_labels[i]

    x_train.append(cv2.resize(img, (im_size, im_size)))

    y_train.append(label)

    i += 1
for f in tqdm(df_test['id'].values):

    img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(f))

    x_test.append(cv2.resize(img, (im_size, im_size)))
y_train_raw = np.array(y_train, np.uint8)

x_train_raw = np.array(x_train, np.float32) / 255.

x_test  = np.array(x_test, np.float32) / 255.

num_class = y_train_raw.shape[1]
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=1)
model.fit(X_train, Y_train, epochs=50, validation_data=(X_valid, Y_valid), verbose=1)

preds = model.predict(x_test, verbose=1)

sub = pd.DataFrame(preds)

# Set column names to those generated by the one-hot encoding earlier

col_names = one_hot.columns.values

sub.columns = col_names

# Insert the column id from the sample_submission at the start of the data frame

sub.insert(0, 'id', df_test['id'])

sub.head(5)
sub.to_csv('submission.csv', index = False)