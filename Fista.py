# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:40:10 2018

@author: Pragya
"""

# FISTA with backtracking

# Objective function is F(x) = f(x) + g(x)
# f(x) = 0.5 * ||Ax - b||**2
# g(x) = lambda * l1norm(x)

import numpy as np
#from lightning.regression import FistaRegressor as FC
from numpy import linalg
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from numpy import genfromtxt
import math
import random



def f(w,X,y): #Squared Loss
    return 0.5 * (linalg.norm(y- X.dot(np.asarray(w).T),2)**2)
    
def gradw(w0,X,y): #for squared loss
    #print (-1) * X.T.dot((y- X.dot(w0)))
    g = (-1) * X.T.dot((y- X.dot(w0)))
    return g
    
def g(w, lambda1):
    return lambda1 * linalg.norm(w,2)
    
def F(w,X,y, lambda1):
    return f(w,X,y) + g(w, lambda1)
   
def S(v, t): #Soft threshold
    return np.sign(t) * np.maximum(np.abs(t) - v, 0)
    
def prox(v,l):
    return (max(0,(linalg.norm(v,2)) - l)/ (linalg.norm(v,2) + np.finfo(np.float32).eps)) * v 
    """
    # returns a vector of size equal to v 
    arr = np.zeros(len(v))
    for i in range(len(v)):
        arr[i] = max(0, (v - l)[i]) - max(0, (-v - l)[i])
    return arr"""
    
def Q(a,b,L,X,y,lambda1):
    return f(b,X,y) + (a - b).T.dot(gradw(b,X,y)) + L/2*linalg.norm(a-b,2) + g(a,lambda1)

def getLf(X):
    return linalg.norm(X)**2
    
def fista_fit(X,y,lambda1,eta):
    
    wk = np.zeros(X.shape[1])
    yk_1 = np.zeros(X.shape[1])
    maxIt = 500
    it = 1
    tk = 1
    L = 1
    #print("Lf is ",L)
    
    while it<=maxIt:
        wk_1 = wk.copy()
        
        
        # find L using backtracking
        Lbar = L; 
        while True: 
            l1 = lambda1/Lbar; 
            zk = prox(yk_1 - 1/Lbar*gradw(yk_1,X,y), l1)
            Fval = F(zk,X,y,l1)
            Qval = Q(zk, wk_1, Lbar,X,y,l1)
            
            if Fval <= Qval:
                break
            Lbar = Lbar*eta; 
            L = Lbar; 
        
        l2 = lambda1/L
        
        wk = prox(yk_1 - 1/L*gradw(yk_1,X,y),l2)
        #print("wk is ", wk)
        """
        L = getLf(X)
        yk = yk - gradw(yk,X,y)/L
        wk = S(1/L, yk)
        """        
        tk_plus_1 = (1. + math.sqrt(1. + 4. * tk ** 2)) / 2.
        yk = wk + ((tk - 1.) / tk_plus_1) * (wk - wk_1) 
        
        #print "iteration ", it, " at w = ", wk," Objective function value ", F(wk,X,y)
        
        #if(((wk_1 - wk).T.dot(wk_1 - wk)) <= 1e-5):
        if((linalg.norm(wk - wk_1,1)/len(wk)) <= 1e-8):       
            #print("I have converged")
            break        
        
        it= it+1
        #for next iteration
        tk = tk_plus_1
        yk_1 = yk
        
    return wk
 
def fista_predict(w, Xtest, Ytest):
    Ypredicted= []
    acc = 0
    for i in range(len(Xtest)):
        s = np.sign(np.asarray(Xtest[i]).T.dot(w))
        Ypredicted.append(s)
        if(s == Ytest[i]):
            acc = acc +1
    #print(Ypredicted)
    print(acc * 100/len(Ytest))   
    return Ypredicted
    
   
if __name__ == '__main__':
    
    """ 
    irisData = datasets.load_iris()
    Xr = irisData.data[:, :] 
    Y = []
    for target in irisData.target:
        if(target==0):
            Y.append(-1)
        else:
            Y.append(target)
    """
    dataset = (genfromtxt('C:\Users\Pragya\Desktop\Datasets\PimaIndians.csv', delimiter=','))
    #print(dataset)
    
    Xr=[]
    Y=[]
    for data in dataset:
        arr=[]
        for val in data[:8]:
            arr.append(float(val))
        Xr.append(arr)
        fx = int(data[8:9])
        if(fx==0):
            Y.append(-1)
        else:
            Y.append(1)
    Xr[0][0] = 6.0  
    """
    dataset = (genfromtxt('C:\Users\Pragya\Desktop\Datasets\IrisData.csv', delimiter=','))
    #print(dataset)
    
    Xr=[]
    Y=[]
    for data in dataset:
        arr=[]
        for val in data[:4]:
            arr.append(float(val))
        Xr.append(arr)
        fx = int(data[4:5])
        Y.append(fx)
    Xr[0][0] = 5.1    
    
    dataset = (genfromtxt('C:\Users\Pragya\Desktop\Datasets\Australian.csv', delimiter=','))
    #print(dataset)
    
    Xr=[]
    Y=[]
    for data in dataset:
        arr=[]
        for val in data[:14]:
            arr.append(float(val))
        Xr.append(arr)
        fx = int(data[14:15])
        if(fx==0):
            Y.append(-1)
        else:
            Y.append(1)
    Xr[0][0] = 1.0              
    """    
    # 70% data for training, remaining for testing
    X, Xtest, y, Ytest = train_test_split(Xr, Y, test_size=0.3, random_state=0)
    X = np.asarray(X)
    y = np.asarray(y)
    
    lambda1 =0.01
    eta = 1.5
    
    w = fista_fit(X,y,lambda1,eta)
    print("w is ", w)
    print("Accuracy in my FISTA implementation: ")
    fista_predict(w,Xtest,Ytest)
    
    '''
    #Verifying Output using Fista defined in lightning module
    classi = FC(C=0.5,alpha=lambda1,max_steps=500,penalty='l1')
    classi.fit(X,y)
    
    CompPred = []
    predicted = classi.predict(Xtest)
    acc = 0
    for i in range(len(predicted)):
        s = np.sign(predicted[i])
        CompPred.append(s)
        if(s==Ytest[i]):
            acc = acc+ 1
    print("Accuracy in Lightning's FISTA implementation: ")    
    print(acc*100/len(predicted))
    
    '''
        #Adding Outliers into 10% of the data
    
    print
    print
    print("Following are the results with 10% false data")
    print
    
    for i in range(int(0.1*len(y))):
        r = random.randint(0,len(y)-1)
        if(y[r]==1):
            y[r] = -1
        else:
            y[r] = 1
            
    for i in range(int(0.1*len(Ytest))):
        r = random.randint(0,len(Ytest)-1)
        if(Ytest[r]==1):
            Ytest[r] = -1
        else:
            Ytest[r] = 1
    
    w = fista_fit(X,y,lambda1,eta)
    print("Accuracy in my FISTA implementation: ")
    fista_predict(w,Xtest,Ytest)
    
    '''
    #Verifying Output using Fista defined in lightning module
    classi = FC(C=1.0,alpha=lambda1,max_steps=500,penalty='l1')
    classi.fit(X,y)
    
    CompPred = []
    predicted = classi.predict(Xtest)
    acc = 0
    for i in range(len(predicted)):
        s = np.sign(predicted[i])
        CompPred.append(s)
        if(s==Ytest[i]):
            acc = acc+ 1
    print("Accuracy in Lightning's FISTA implementation: ")    
    print(acc*100/len(predicted))'''