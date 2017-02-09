import pandas as pd 
from sklearn import preprocessing,cross_validation 
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
data = pd.read_csv('glass.csv')

print(data.head(5))
print(data.info())
#print(data['Fe'])
print(data.describe())
Y = data['Type']
data.drop(['Type','Ba'],1,inplace= True)


data= preprocessing.scale(data)
X_Train ,X_Test, Y_Train, Y_Test = cross_validation.train_test_split(data,Y,test_size=.2)

model=SVR(kernel='linear')
selector = RFE(model, 3, step=1)
selector = selector.fit(X_Train, Y_Train)
print(selector.support_)
print(selector.ranking_)
predict = selector.predict(X_Test)
score= selector.score(X_Test,Y_Test)
print(score)
print(predict)


