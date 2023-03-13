# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from textblob import TextBlob

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train.tail()
selected_text = []

for text in test['text']:

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity

    if polarity == 0:

        selected_text.append(text)

    else:

        selected_text.append(' '.join(blob.sentiment_assessments.assessments[0][0]))
result = pd.DataFrame()

result['textID'] = test['textID']

result['selected_text'] = selected_text

result.to_csv('submission.csv',index=False)
result.head()