import numpy as np 
from sklearn import datasets ,cross_validation ,preprocessing 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

dataset = datasets.load_digits()
X_train, X_test, Z_train, Z_test = cross_validation.train_test_split(dataset.data, dataset.target, test_size = 0.2)

for model,name in [(LinearRegression(),'LR'),(KNeighborsClassifier(n_neighbors=3),'KNN'),(GaussianNB(),'NB')]:
	model.fit(X_train,Z_train)
	print(name,model.score(X_test,Z_test))

