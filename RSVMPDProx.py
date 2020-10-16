

# PD Prox with rescaled alpha -Hinge loss

from FISTA import fista_fit
import numpy as np
from numpy import linalg
from sklearn.cross_validation import train_test_split
from numpy import genfromtxt
import random
import pandas as pd
# Primal Dual Prox method for Non Smooth Optimization
def F(w,alpha):
    return f(w,alpha) + lambda1*g(w)

def f(w, alpha):
    [n, d] = X.shape
    sum = 0
    for i in range(n):
        yi = y[i]
        xi = X[i]
        alphai = alpha[i]
        sum = sum + RescaledHingeloss(alphai, w, xi, yi,etaR)
    return 1./n * sum
    
def RescaledHingeloss(alphai, w, xi, yi, etaR):
    beta = 1/(1-np.exp(-1*etaR))
    val = (-1) * etaR * alphai * (1 - yi*(w.dot(xi.T)))
    ans = beta * (1 - np.exp(val))
    return ans
    
def gradienta(w0, alpha0):
    beta = 1/(1-np.exp(-1*etaR))
    [n, d] = X.shape
    sum = np.zeros(n)
    for i in range(n):
        yi = y[i]
        xi = X[i]
        vali = 1 - yi*(w0.dot(xi.T))
        
        sum[i] = beta * etaR * vali * np.exp(-1*etaR*alpha0[i]*vali)
    return 1./n * (sum)
    
def gradientw(w0,alpha0):
    n = X.shape[0]
    beta = 1/(1-np.exp(-1*etaR))
    sum = 0
    for i in range(n):
        yi = y[i]
        xi = X[i]        
        vali = np.exp(-1 * etaR * alpha0[i]* (1 - yi*(w0.dot(xi.T))))
        sum  = sum + (alpha0[i] * yi * xi * vali)
    return -(beta * etaR)/n * sum
    
    
def g(w):
    return linalg.norm(w,1)
    
def getStep():
    """
    R = linalg.norm(X[random.randint(1,n-1)])
    print "R is ", R
    return R/n"""
    
    Norm = np.zeros(n)
    for i in range(n):
        Norm[i] = linalg.norm(X[i],2)
        
    R = linalg.norm(Norm, np.inf)
    print ("R is ", R)
    c = R**2/n
    print("c is ", c)
    return np.sqrt(0.5/c)
    
    
    
def Proj(zbarV):
    arr = np.zeros(n)
    for i in range(n):
        zbar = zbarV[i]
        #we know that Qa is [0,1]
        if(zbar>=0 and zbar<=1):
            arr[i] = zbar
        elif(zbar<0):
            arr[i] = 0
        else:
            arr[i] = 1 
    return arr

def argmin(val, gamma):
    Xtr = np.identity(d)
    Ytr = val
    lam = gamma*lambda1
    return fista_fit(Xtr, Ytr, lam, 1.5)
        
   
def PDProxDual_train(gamma,n,d):
    T = 10
    wPrev = np.zeros(d)
    betaPrev = np.zeros(n)
    
    alpha = np.zeros(n)
    storealpha = np.zeros((T,n))
    beta = np.zeros(n)
    w = np.zeros(d)
    storew = np.zeros((T,d))
    
    
    for t in range(0,T):
        #print gradienta(wPrev,betaPrev)
        alpha = Proj(betaPrev + gamma * gradienta(wPrev,betaPrev))
        storealpha[t] = alpha
        #print "a is ", alpha
        #print "gradw is ", gradientw(wPrev, alpha)
        val = wPrev - gamma * gradientw(wPrev, alpha) 
        #print "val is", val
        w = argmin(val, gamma)
        #print "w is ", w
        storew[t] = w
        #print betaPrev + gamma*gradienta(w, alpha)
        beta = Proj(betaPrev + gamma*gradienta(w, alpha))
        #print "beta is ", beta
        
        wPrev = w
        betaPrev = beta
        
    wt = sum(storew)/T
    at = sum(storealpha)/T
    
    return wt, at
    
def PDProxDual_predict(w, Xtest, Ytest):
    Ypredicted= []
    acc = 0
    l = np.asarray(Xtest).shape[0]
    for i in range(l):
        s = np.sign(np.asarray(Xtest[i]).T.dot(w))
        Ypredicted.append(s)
        if(s == Ytest[i]):
            acc = acc +1
    #print(Ypredicted)
    print(acc * 100/l)   
    return Ypredicted
        
        
dataset = pd.read_csv(r'C:\Users\nwoslab2\Downloads\PDProx\CovType.csv', delimiter=',')
#print(dataset)
len = dataset.shape[1]

Xr=[]
Y=[]
for data in dataset:
    arr=[]
    for val in data[:len-1]:
        arr.append(float(val))
    Xr.append(arr)
    fx = int(data[len-1:len])
    if(fx==1 or fx ==2 or fx ==3):
        Y.append(-1)
    else:
        Y.append(1)
Xr[0][0] = 2596            

""" 70% data for training, remaining for testing"""
X, Xtest, y, Ytest = train_test_split(Xr, Y, test_size=0.3, random_state=0)
X = np.asarray(X)
y = np.asarray(y)     

#Regularization Parameter
lambda1 = 0.001

#Rescaling factor
etaR = 0.2

[n, d] = X.shape

gamma = getStep()
print ("Step size is ", gamma)
wt, at = PDProxDual_train(gamma,n,d) 
#print "wt is ", wt, "alpha is ", at

print ("Accuracy is: ")
PDProxDual_predict(wt, Xtest, Ytest)

print
print("Following are the results with 10% false data")
print

leny = np.asarray(y).shape[0]
for i in range(int(0.1*leny)):
    r = random.randint(0,leny-1)
    if(y[r]==1):
        y[r] = -1
    else:
        y[r] = 1

lenYtest = np.asarray(Ytest).shape[0]       
for i in range(int(0.1*lenYtest)):
    r = random.randint(0,lenYtest-1)
    if(Ytest[r]==1):
        Ytest[r] = -1
    else:
        Ytest[r] = 1


wt, at = PDProxDual_train(gamma,n,d) 
#print "wt is ", wt, "alpha is ", at

print ("Accuracy is: ")
PDProxDual_predict(wt, Xtest, Ytest)   