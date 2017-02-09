
import numpy as np 
import matplotlib.pyplot as plt 
#/
#X=np.random.uniform(-4,4,500)
#Y=X+np.random.standard_normal(500)+2.5
#plt.plot(x,y,'o')
#plt.show()

def dg(X,Y,alpha,a0,a1): 
	temp0=((1/len(Y))*np.sum(a0+np.dot(a1,X)-Y))
	temp1=((1/len(Y))*np.sum(a0+a1*X-Y))*X
	ao = a0-alpha*temp0
	a1 = a1-alpha*temp1
	return a0,a1,temp0

X = np.array([1,2,3,4,5,6])
Y = np.array([1.5,4,4,6, 7,8])
plt.figure(1)
plt.plot(X,Y,'o')

a0=0
a1=0
l =[]
a0_list=[]
a1_list=[]
alpha=0.01

for i in range(1000):
	temp0=alpha*((2/len(Y))*np.sum(a0+a1*X-Y))
	temp1=alpha*((1/len(Y))*np.sum(a0+a1*X-Y))*X
	a0 = a0-temp0
	a1 = a1-temp1

	#plt.plot(X,a0+X*a1)
	a0_list.append(a0)
	a1_list.append(a1)
	l.append(temp0)

#	print(a0)
#a0_list= np.array(a0_list)
#a1_list = np.array(a1_list)
print(a0,a1)
plt.plot(X,a0+X*a1)
#plt.figure(2)
#plt.plot(X,a0_list+X.a1_list)
plt.figure(3)
plt.plot(l)
plt.show()
#plt.show()