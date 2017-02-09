import numpy 
from sklearn import datasets ,cross_validation ,preprocessing 
from sklearn.linear_model import LinearRegression
dataset = datasets.load_iris()
X=dataset.data
X = preprocessing.scale(X)
Y=dataset.target
print(len(X))
#print(X)
#print(Y)
X_Train ,X_Test, Y_Train, Y_Test = cross_validation.train_test_split(X,Y,test_size=.2)
model = LinearRegression()
model.fit(X_Train,Y_Train)
b= model.predict(X_Test)
score = model.score(X_Test,Y_Test)
print(score)
print(b)
print(Y_Test)
print(model.coef_)