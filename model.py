import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


data1  = pd.read_csv('../data/creditcard.csv')

#clear null
data1.isnull().sum().max()

#check columns


#fine detail
print('No Frauds', round(data1['Class'].value_counts()[0]/len(data1) * 100,2), '% of the dataset')
print('Frauds', round(data1['Class'].value_counts()[1]/len(data1) * 100,2), '% of the dataset')

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data1['scaled_amount'] = rob_scaler.fit_transform(data1['Amount'].values.reshape(-1,1))
data1['scaled_time'] = rob_scaler.fit_transform(data1['Time'].values.reshape(-1,1))
data1.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = data1['scaled_amount']
scaled_time = data1['scaled_time']

data1.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data1.insert(0, 'scaled_amount', scaled_amount)
data1.insert(1, 'scaled_time', scaled_time)


print(data1.head)

x = data1.drop('Class',axis=1)
y = data1['Class']

stratkFold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in stratkFold.split(x, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = x.iloc[train_index], x.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
    
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

#check if test and train data have similar spread
print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

