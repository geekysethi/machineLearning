
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

X1 = np.array([2104,1416,1534,852]) #size(feet**2)
X2 = np.array([5,3,3,2])           #Numbers of bedrooms
X3 = np.array([1,2,2,1])		#NO. of floors
X4 = np.array([45,40,30,36])  		#Age of home 
Y = np.array([460,232,315,178])  #price
plt.figure(1)
#plt.plot(X,Y,'o')

a0=0
a1=0
a2=0
a3=0
a4=0
l =[]
a0_list=[]
a1_list=[]
alpha=0.03

for i in range(100):
	temp0=alpha*((1/len(Y))*np.sum(a0+a1*X1-Y))
	temp1=alpha*((1/len(Y))*np.sum(a0+a1*X1-Y))*X1
	temp2=alpha*((1/len(Y))*np.sum(a0+a1*X2-Y))*X2
	temp3=alpha*((1/len(Y))*np.sum(a0+a1*X3-Y))*X3
	temp4=alpha*((1/len(Y))*np.sum(a0+a1*X4-Y))*X4
	

	a0 = a0-temp0
	a1 = a1-temp1
	a2 = a2-temp2
	a3 = a3-temp3
	a4 = a4-temp4

	#plt.plot(X,a0+X*a1)
	a0_list.append(a0)
	a1_list.append(a1)
	l.append(temp0)

#	print(a0)
#a0_list= np.array(a0_list)
#a1_list = np.array(a1_list)
print(a0,a1)
plt.plot(X1,a0+X1*a1+X2*a2+X3*a3+X4*a4)
#plt.figure(2)
#plt.plot(X,a0_list+X.a1_list)
plt.figure(3)
plt.plot(l)
plt.show()
#plt.show()