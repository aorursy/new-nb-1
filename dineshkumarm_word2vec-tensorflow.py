# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
import nltk as nl

train['tokens'] = [nl.word_tokenize(sentences) for sentences in train.text]

words = []

for item in train.tokens:

    words.extend(item)



stemmer = nl.stem.lancaster.LancasterStemmer()

words = [stemmer.stem(word) for word in words]





filtered_words = [word for word in words if word not in nl.corpus.stopwords.words('english')]







import gensim

# let X be a list of tokenized texts (i.e. list of lists of tokens)

model = gensim.models.Word2Vec(filtered_words, size=100)

w2v = dict(zip(model.wv.index2word, model.wv.syn0))
i = 0

for index,item in train.iterrows():

    if(i < len(item['tokens'])):

        i = len(item['tokens'])

print(i)
training = []

i = 0

for index,item in train.iterrows():

    vec = np.zeros(100)

    token_words = [stemmer.stem(word) for word in item['tokens']]

    token_words = [word for word in token_words if word not in nl.corpus.stopwords.words('english')]

    for w in token_words:

        if w in w2v:

            vec += w2v[w]

    norm = np.linalg.norm(vec)

    if norm != 0:

        vec /= np.linalg.norm(vec)

    

    

    training.append([vec,item['author']])
training_new = np.array(training)
from numpy import array



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(training_new[:,1])



# binary encode

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)



train_y = onehot_encoded
train_x = list(training_new[:,0])

import tensorflow as tf

import tflearn

# reset underlying graph data

tf.reset_default_graph()

# Build neural network

net = tflearn.input_data(shape=[None, len(train_x[0])])

net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')

net = tflearn.regression(net)

 

# Define model and setup tensorboard

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training (apply gradient descent algorithm)

model.fit(train_x, train_y, n_epoch=10, batch_size=8, show_metric=True)

model.save('model.tflearn')
test = pd.read_csv('../input/test.csv')

test.head()
test['tokens'] = [nl.word_tokenize(sentences) for sentences in test.text]

testing = []

for index,item in test.iterrows():

    vec = np.zeros(100)

    token_words = [stemmer.stem(word) for word in item['tokens']]

    token_words = [word for word in token_words if word not in nl.corpus.stopwords.words('english')]

    for w in token_words:

        if w in w2v:

            vec += w2v[w]

    norm = np.linalg.norm(vec)

    if norm != 0:

        vec /= np.linalg.norm(vec)

    

    

    testing.append(vec)
testing = list(np.array(testing))

predicted = model.predict(X=testing)

result_val = pd.DataFrame(predicted)

result_val.columns = ["EAP","HPL","MWS"]

result = pd.DataFrame(columns=['id'])

result['id'] = test['id']

result['EAP'] = result_val['EAP']

result['HPL'] = result_val['HPL']

result['MWS'] = result_val['MWS']

result.head()