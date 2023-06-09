# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt




import seaborn as sns

sns.set()



from IPython.display import HTML



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

# Any results you write to the current directory are saved as output.
#reading data 

train = pd.read_csv("../input/train.csv", nrows=10000000,

                   dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

train.head(5)



train.rename({"acoustic_data": "signal","time_to_failure":"quaktime"})

#Exploratory analysis
