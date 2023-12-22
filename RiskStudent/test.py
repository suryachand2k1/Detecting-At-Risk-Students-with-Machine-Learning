


#Distinction = 0, Fail = 1, Pass=2,withdrawn = 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('Dataset/OULAD.csv')
dataset.drop(['id'], axis = 1,inplace=True)
dataset.fillna(0, inplace = True)

print(dataset.final_result.unique()) 

le = LabelEncoder()
dataset['code_module'] = pd.Series(le.fit_transform(dataset['code_module']))
dataset['code_presentation'] = pd.Series(le.fit_transform(dataset['code_presentation']))
dataset['assessment_type'] = pd.Series(le.fit_transform(dataset['assessment_type']))
dataset['gender'] = pd.Series(le.fit_transform(dataset['gender']))
dataset['region'] = pd.Series(le.fit_transform(dataset['region']))
dataset['highest_education'] = pd.Series(le.fit_transform(dataset['highest_education']))

dataset['imd_band'] = pd.Series(le.fit_transform(dataset['imd_band']))
dataset['age_band'] = pd.Series(le.fit_transform(dataset['age_band']))
dataset['disability'] = pd.Series(le.fit_transform(dataset['disability']))
dataset['final_result'] = pd.Series(le.fit_transform(dataset['final_result']))

dataset = dataset.values
cols = dataset.shape[1] - 1
X = dataset[:,0:cols]
Y = dataset[:,cols]
X = normalize(X)
print(X)
print(Y)

(unique, counts) = np.unique(Y, return_counts=True)
print(unique)
print(counts)
pca = PCA(n_components = 10)
X = pca.fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

rfc = LogisticRegression(max_iter=500)
rfc.fit(X_train, y_train)
predict = rfc.predict(X_test)
acc = accuracy_score(y_test,predict) * 100
print(acc)








    
