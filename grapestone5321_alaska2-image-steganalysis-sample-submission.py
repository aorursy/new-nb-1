# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input'))





# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')

df.to_csv('submission.csv', index=False)
df.head(10)