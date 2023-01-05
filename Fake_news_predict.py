import numpy as np
import  pandas as pd

#import data
data = pd.read_csv('news.csv')

#ecoding
X = data['text']
y = data['label']
target_dict ={'FAKE':0,'REAL':1}
y  = y.map(target_dict)

#slpit
from  sklearn.model_selection import  train_test_split
X_train , X_test , y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#convert to train
from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec= vector.fit_transform(X_train)
X_test_vec= vector.transform(X_test)


#train
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

pac=PassiveAggressiveClassifier()
pac.fit(X_train_vec,y_train)
#Test accuracy

y_pred=pac.predict(X_test_vec)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
