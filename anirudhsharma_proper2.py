import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df=pd.read_csv("../input/train.csv")
print(df)
df2=df.place_id
print(df2)
