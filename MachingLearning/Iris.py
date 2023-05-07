import pandas as pd
import numpy as np
import pickle

df = pd.read_excel('Iris.xlsx')

from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['variety'])
Y = df['variety']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# model training
model.fit(x_train, y_train)

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

print(model.fit(x_train, y_train))

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)

# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
print(model.fit(x_train, y_train))

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)

# save the model
import pickle
filename = 'IRISdmodel.sav'
pickle.dump(model,open(filename,'wb'))

load_model = pickle.load(open(filename,'rb'))
p= load_model.predict([[6.0,2.2,4.0,1.0]])
print(p)