import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


import sys
sys.version
def df_load(file, batch_size, skip = 1):

    ''' Build dataframe by iterating over chunks. Option to skip chunks and

        therefore read in less data. '''



    reader = pd.read_csv(file, chunksize=batch_size,

                         dtype=np.float16)



    df = pd.concat((chunk for i, chunk in enumerate(reader) if i % skip == 0))



    return df
input_path = '../input/'

df_num = df_load(input_path+'train_numeric.csv',

                 batch_size=100000)
df_num.head()
np.random.seed(117)

num_samples = 200

random_picks = (np.array(np.random.random(num_samples)) * df_num.shape[1]).astype(int)

random_picks = sorted(random_picks)

print(random_picks)
for feature in random_picks[:150]:

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot linear scale feature density

    df_num.iloc[:, feature][(df_num.Response == 0)].dropna().hist(bins=50, alpha=0.4, normed=True, ax=axes[0],

                                                         label='%d response 0 samples' % len(df_num.iloc[:, feature][(df_num.Response == 0)].dropna()))

    df_num.iloc[:, feature][(df_num.Response == 1)].dropna().hist(bins=50, alpha=0.4, color='r', normed=True, ax=axes[0],

                                                         label='%d response 1 samples' % len(df_num.iloc[:, feature][(df_num.Response == 1)].dropna()))

    # Plot log scale feature density

    df_num.iloc[:, feature][(df_num.Response == 0)].dropna().hist(bins=50, alpha=0.4, normed=True, ax=axes[1], log=True)

    df_num.iloc[:, feature][(df_num.Response == 1)].dropna().hist(bins=50, alpha=0.4, color='r', normed=True, ax=axes[1], log=True)



    axes[0].legend(loc='upper left')

    plt.suptitle('Feature column "%s" histogram' % df_num.columns[feature], fontsize=15)

    plt.show()