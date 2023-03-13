import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



import keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers



import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
## Train the model 

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_noemb_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1:.4f}, Precision is {2:.4f} and Recall is {3:.4f}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_noemb_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)
del model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
## Train the model 

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_noemb_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_noemb_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)
del model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_glove_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_glove_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_glove_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_glove_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_glove_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_fasttext_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_fasttext_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_fasttext_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_fasttext_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_paragram_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_paragram_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_paragram_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
#Train/Test loss v/s Epoch

data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})

ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)

ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')

sns.despine()

plt.show()
pred_paragram_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax

import gc; gc.collect()

time.sleep(10)
pred_val_y_gru = 0.34*pred_glove_val_y_gru + 0.33*pred_fasttext_val_y_gru + 0.33*pred_paragram_val_y_gru 

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_val_y_gru>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_val_y_gru>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_val_y_gru>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_val_y_gru>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
pred_val_y_lstm = 0.34*pred_glove_val_y_lstm + 0.33*pred_fasttext_val_y_lstm + 0.33*pred_paragram_val_y_lstm 

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_val_y_lstm>thresh).astype(int))))
#Plot of F1 score, Precision and Recall for different thresholds 

f1s = []

precisions = []

recalls = []

    

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(val_y, (pred_val_y_lstm>thresh).astype(int))

    precision = metrics.precision_score(val_y, (pred_val_y_lstm>thresh).astype(int))

    recall = metrics.recall_score(val_y, (pred_val_y_lstm>thresh).astype(int))



    f1s.append(f1)

    precisions.append(precision)

    recalls.append(recall)



    data = pd.DataFrame(data = {

        'F1': f1s,

        'Precision': precisions,

        'Recall': recalls})

sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)

sns.despine()

plt.show()
pred_test_y_gru = 0.34*pred_glove_test_y_gru + 0.33*pred_fasttext_test_y_gru + 0.33*pred_paragram_test_y_gru

pred_test_y_gru = (pred_test_y_gru>0.35).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y_gru

out_df.to_csv("submission.csv", index=False)