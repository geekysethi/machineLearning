import pandas as pd 
from sklearn import preprocessing,cross_validation 
from sklearn.feature_selection import RFE,SelectKBest,chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR ,SVC
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('autos.csv',encoding= 'cp1252')

print(data.head(5))

data.drop(['dateCrawled','dateCreated','nrOfPictures','postalCode','lastSeen','abtest' ],1,inplace=1)
print(data.head(5))

print(data.info())
for col in data.columns:
	print(col,data[col].isnull().sum())
print('***************************************')
data['notRepairedDamage'].fillna('Nein',inplace = True)
data['gearbox'].fillna('manuell',inplace = True)
data['vehicleType'].fillna('limousine',inplace = True)
data['model'].fillna('golf',inplace = True)
data['fuelType'].fillna(' benzin',inplace = True)

for col in data.columns:
	print(col,data[col].isnull().sum())

#print(data['fuelType'].mode())
columns =[]
for i in data.columns:
	if (type(i)=='object'):
		columns.append(i)
print(columns)
'''
le = preprocessing.LabelEncoder()
for col in data.columns:
	data[col]= le.fit_transform(data[col])

print(data.head(5))
print('-------------------------------')

target = data['price']
data=preprocessing.scale(data)
target = preprocessing.scale(target)
X_Train ,X_Test, Y_Train, Y_Test = cross_validation.train_test_split(data,target,test_size=.2)
print('Training Started ')

model = LinearRegression()
model.fit(X_Train,Y_Train)
score = model.score(X_Test,Y_Test)
print(score)

'''


#selector = RFE(model, 3, step=1)
#selector = selector.fit(X_Train, Y_Train)
#print(selector.support_)
#print(selector.ranking_)
#print(data.info())
#print(data.describe())

#target = data['price']
#data.drop('price',1,inplace=True)

#X_Train ,X_Test, Y_Train, Y_Test = cross_validation.train_test_split(data,target,test_size=.2)






'''


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

model=SVR(kernel='linear')
selector = RFE(model, 3, step=1)
selector = selector.fit(X_Train, Y_Train)

print(selector.support_)
print(selector.ranking_)
E=[]
P=[]
for i in range(len(data)):
	if(target=)







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