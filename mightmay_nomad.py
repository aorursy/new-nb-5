# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
X = train.drop(['id','bandgap_energy_ev','formation_energy_ev_natom'], axis=1)
Yf = (train['formation_energy_ev_natom'])
Yb = (train['bandgap_energy_ev'])

regr = RandomForestRegressor(max_depth=15, n_estimators=500)
regr.fit(X, Yf)
Pf = regr.predict(test.drop(['id'], axis=1))

clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01,max_depth=15, loss='ls')
clf.fit(X, Yb)
Pb = clf.predict(test.drop(['id'], axis=1))
sub=pd.read_csv("../input/sample_submission.csv")
sub["formation_energy_ev_natom"]=Pf
sub["bandgap_energy_ev"]=Pb
print(sub)
sub.to_csv("output.csv",index=False)
# Any results you write to the current directory are saved as output.