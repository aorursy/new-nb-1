import seaborn as sns



sns.lineplot(data=grouped_sel, x='date', y='total_sell', hue='product_sku');
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot
pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame

pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame
df_product = pd.read_csv('/kaggle/input/uisummerschool/Product.csv')

df_online = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')

df_offline = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')

df_marketing = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')
df_marketing.head()
df_marketing.columns = ['date', 'offline_spend', 'online_spend']
df_product.head()
df_product.columns = ['product_sku', 'stock_code']
df_offline.head()
df_offline.columns = ['inv_no', 'date', 'stock_code', 'quantity']
df_offline.head()
df_online.head()
df_online.columns = ['transacion_id', 'date', 'product_sku', 'product', 'prod_cat', 'quantity', 'avg_price', 'revenue', 'tax', 'delivery']
df_online.head()
grouped = df_online.groupby(['date', 'product_sku'])['quantity'].sum().reset_index(name='total_sell')
grouped.head()
grouped['date'] = grouped['date'].astype(str)

grouped['date'] = pd.to_datetime(grouped['date'])
prods = list(grouped['product_sku'].sample(5))
dates = [x.strftime('%Y-%m-%d') for x in pd.date_range('2017-01-01', periods= 30)]
grouped_sel = grouped[(grouped['product_sku'].isin(prods)) & (grouped['date'].isin(dates))]
grouped_sel.info()