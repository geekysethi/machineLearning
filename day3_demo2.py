import pandas as pd 
from sklearn import preprocessing,cross_validation 
from sklearn.feature_selection import RFE,SelectKBest,chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR ,SVC
import matplotlib.pyplot as plt

data = pd.read_csv('mushrooms.csv')

print(data.head(5))

print(data.info())


le = preprocessing.LabelEncoder()
for col in data.columns:
	data[col]= le.fit_transform(data[col])

 
#print(data.head(5))
#print(data.info())
#print(data.describe())
target =data['class']
data.drop(['veil-type','class'],1,inplace = True)
print(data.describe())

data= preprocessing.scale(data)
X_Train ,X_Test, Y_Train, Y_Test = cross_validation.train_test_split(data,target,test_size=.2)

print('Training start')
#model=SVR(kernel='linear')
#model=SVC()
model= KNeighborsClassifier(n_neighbors=17)
model.fit(X_Train,Y_Train)
predict =model.predict(X_Test) 
score = model.score(X_Test,Y_Test)
print(predict)
print(score)
X_new = SelectKBest(chi2, k=2).fit_transform(X_Train, Y_Train)
print(X_new.shape)




'''

selector = RFE(model, 3, step=1)
selector = selector.fit(X_Train, Y_Train)
print(selector.support_)
print(selector.ranking_)
predict = selector.predict(X_Test)
score= selector.score(X_Test,Y_Test)
print(score)
print(predict)

print('SVR')
clf = SVR(kernel='linear')
clf.fit(X_Train,Y_Train)
print(clf.predict(X_Test))
print(clf.score(X_Test,Y_Test))


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
'''