# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/application_train.csv")
df_test = pd.read_csv("../input/application_test.csv")
df_POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv")
df_bureau_balance = pd.read_csv("../input/bureau_balance.csv")
df_previous_application = pd.read_csv("../input/previous_application.csv")
df_installments_payments = pd.read_csv("../input/installments_payments.csv")
df_credit_card_balance = pd.read_csv("../input/credit_card_balance.csv")
df_bureau = pd.read_csv("../input/bureau.csv")





