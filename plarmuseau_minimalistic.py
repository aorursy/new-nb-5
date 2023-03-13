import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#1: unreliable
#0: reliable
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test.info()
test['label']='t'
train.info()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
test=test.fillna(' leeg')
train=train.fillna(' leeg')
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(train['text'].values)


classifier = MultinomialNB()
targets = train['label'].values
classifier.fit(counts, targets)
example_counts = count_vectorizer.transform(test['text'].values)
predictions = classifier.predict(example_counts)
pred=pd.DataFrame(predictions,columns=['label'])
pred['id']=test['id']
pred.groupby('label').count()
pred.to_csv('countvect.csv', index=False)
