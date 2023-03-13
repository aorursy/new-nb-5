# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !pip install sentencepiece

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import logging



logging.basicConfig(level=logging.INFO)

import tensorflow_hub as hub 

import tokenization

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
train = pd.read_csv('train.tsv', sep='\t')

test = pd.read_csv('test.tsv', sep='\t')

import tensorflow_hub as hub 

import tokenization

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)



def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



def build_model(bert_layer, max_len=512):

    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)

    net = tf.keras.layers.Dropout(0.2)(net)

    net = tf.keras.layers.Dense(32, activation='relu')(net)

    net = tf.keras.layers.Dropout(0.2)(net)

    out = tf.keras.layers.Dense(5, activation='softmax')(net)

    

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
max_len = 150

train_input = bert_encode(train.Phrase.values, tokenizer, max_len=max_len)

test_input = bert_encode(test.Phrase.values, tokenizer, max_len=max_len)

train_labels = tf.keras.utils.to_categorical(train.Sentiment.values, num_classes=5)
model = build_model(bert_layer, max_len=max_len)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)



train_history = model.fit(

    train_input, train_labels, 

    validation_split=0.2,

    epochs=3,

    callbacks=[checkpoint, earlystopping],

    batch_size=32,

    verbose=1

)

model.load_weights('model.h5')

test_pred = model.predict(test_input)

test_labels = np.argmax(test_pred)

sub = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')

sub['Sentiment'] = np.argmax(test_pred, axis=-1)

sub.to_csv('submission.csv', index=False)

"""# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")

 

# later...

 

# load json and create model

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")

print("Loaded model from disk")

 """





import imp

import keras.backend

import keras.models

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

import pickle

import time

import keras



from keras.datasets import mnist

from keras.models import Model

from keras import optimizers



from matplotlib import cm, transforms



import innvestigate

import innvestigate.applications

import innvestigate.applications.mnist

import innvestigate.utils as iutils

import innvestigate.utils.visualizations as ivis

from innvestigate.utils.tests.networks import base as network_base
# Specify methods that you would like to use to explain the model. 

# Please refer to iNNvestigate's documents for available methods.

methods = ['gradient', 'lrp.z', 'lrp.alpha_2_beta_1', 'pattern.attribution']

kwargs = [{}, {}, {}, {'pattern_type': 'relu'}]
# build an analyzer for each method

analyzers = []



for method, kws in zip(methods, kwargs):

    analyzer = innvestigate.create_analyzer(method, model, **kws)

    analyzer.fit(train_input, batch_size=256, verbose=1)

    analyzers.append(analyzer)
# specify indices of reviews that we want to investigate

test_sample_indices = [156066, 156067, 156068, 156069, 156070, 156071]



test_sample_preds = [None]*len(test_sample_indices)



# a variable to store analysis results.

analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, 150])



for i, ridx in enumerate(test_sample_indices):

    

    x, y = test_input[ridx], test_labels[ridx]



    t_start = time.time()

    x = x.reshape((1, 1, 150, EMBEDDING_DIM))    



    presm = model.predict_on_batch(x)[0] #forward pass without softmax

    prob = model.predict_on_batch(x)[0] #forward pass with softmax

    y_hat = prob.argmax()

    test_sample_preds[i] = y_hat

    

    for aidx, analyzer in enumerate(analyzers):



        a = np.squeeze(analyzer.analyze(x))

        a = np.sum(a, axis=1)



        analysis[i, aidx] = a

    t_elapsed = time.time() - t_start

    print('Review %d (%.4fs)'% (ridx, t_elapsed))
