# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
df_train = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.
r = np.arange(df_train['time'].min(), df_train['time'].max(), 7*24*60)
c = pd.cut( df_train['time'], r)
g = df_train.groupby(c)
g.describe()

    
    
gb1 = df_train.groupby(['place_id'])
gb1
