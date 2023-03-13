import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd



df = pd.read_csv("/kaggle/input/spooky-author-identification/train.csv")

df.head()
text = []



for row in df["text"][df["author"] == "EAP"]:

    text.append(str(row))

    

corpusEAP = " ".join(text)

corpusEAP = corpusEAP[0:100000]
len(corpusEAP)
import spacy

nlp = spacy.load('en',disable=['parser', 'tagger','ner']) # only for tokenisation

def separate_punc(doc_text):

    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


tokens = separate_punc(corpusEAP)

len(tokens)
# organize into sequences of tokens

train_len = 25+1



# Empty list of sequences

text_sequences = []



for i in range(train_len, len(tokens)):

    

    # Grab train_len# amount of characters

    seq = tokens[i-train_len:i]

    

    # Add to list of sequences

    text_sequences.append(seq)
print (' '.join(text_sequences[0]))

print (' '.join(text_sequences[1]))
len(text_sequences)
from keras.preprocessing.text import Tokenizer

# integer encode sequences of words

tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences)

print (sequences[0])
print (tokenizer.index_word)

print ()

print (" --------------------- ")

print (len(tokenizer.word_counts))
sequences = np.array(sequences)

sequences[0]
X = sequences[:,:-1]

y = sequences[:,-1]
print (X.shape, y.shape)

import keras

from keras.models import Sequential

from keras.layers import Dense,LSTM,Embedding



vocabulary_size = len(tokenizer.word_counts)

vocabulary_size = vocabulary_size + 1

seq_len = X.shape[1]



model = Sequential()

model.add(Embedding(vocabulary_size, 25, input_length=seq_len))

model.add(LSTM(150, return_sequences=True)) # to stack LSTM we need return seq 

model.add(LSTM(150))

model.add(Dense(150, activation='relu'))



model.add(Dense(vocabulary_size, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
model_test = Sequential()

model_test.add(Embedding(input_dim = vocabulary_size, output_dim = 2, input_length=seq_len))

model_test.compile('rmsprop', 'mse')

output_array = model_test.predict(X)

print (output_array.shape)

out = pd.DataFrame(output_array[0])

out.head()

from keras.utils import to_categorical

y = to_categorical(y, num_classes=vocabulary_size)

# fit model

model.fit(X, y, batch_size=256, epochs=100,verbose=1)
from keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):

    '''

    INPUTS:

    model : model that was trained on text data

    tokenizer : tokenizer that was fit on text data

    seq_len : length of training sequence

    seed_text : raw string text to serve as the seed

    num_gen_words : number of words to be generated by model

    '''

    

    # Final Output

    output_text = []

    

    # Intial Seed Sequence

    input_text = seed_text

    

    # Create num_gen_words

    for i in range(num_gen_words):

        

        # Take the input text string and encode it to a sequence

        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        

        # Pad sequences to our trained rate 

        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

        

        # Predict Class Probabilities for each word

        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]

        

        # Grab word

        pred_word = tokenizer.index_word[pred_word_ind] 

        

        # Update the sequence of input text (shifting one over with the new word)

        input_text += ' ' + pred_word

        

        output_text.append(pred_word)

        

    # Make it look like a sentence.

    return ' '.join(output_text)
text_sequences[100]


seed_text = ' '.join(text_sequences[100])

generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=10)
df.head()
import random



df_augumented = pd.DataFrame(columns=['text', 'author'])



for i in range (200):

    random_pick = random.randint(0,len(text_sequences))

    seed_text = ' '.join(text_sequences[random_pick])

    text = generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=20)

    df_augumented["text"]

    df_augumented = df_augumented.append({'text': text, "author" : "EAP"}, ignore_index=True)

    

df_augumented.head()    

    
data_prev = df[["text", "author"]]

data = pd.concat([data_prev, df_augumented], axis = 0)

data.shape
data['author_num'] = data["author"].map({'EAP':0, 'HPL':1, 'MWS':2})
X = data['text']

y = data['author_num']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words = 'english')

X_train_matrix = vect.fit_transform(X_train) 

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(X_train_matrix, y_train)

print(clf.score(X_train_matrix, y_train))

X_test_matrix = vect.transform(X_test) 

print (clf.score(X_test_matrix, y_test))

predicted_result=clf.predict(X_test_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test,predicted_result))
X = data['text']

y = data['author_num']



from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words = 'english')

X_train_matrix = vect.fit_transform(X) 

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(X_train_matrix, y)

test = pd.read_csv("/kaggle/input/spooky-author-identification/test.csv")

test_matrix = vect.transform(test["text"])

predicted_result = clf.predict_proba(test_matrix)

result=pd.DataFrame()

result["id"]=test["id"]

result["EAP"]=predicted_result[:,0]

result["HPL"]=predicted_result[:,1]

result["MWS"]=predicted_result[:,2]

result.head()

result.to_csv("submission_v5.csv", index=False)