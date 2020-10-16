
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
from statistics import mean


from RSVMPDProx import PDProxDual_train, PDProxDual_predict,PDProxDual_SupportVectorRatio


#arr = [76,80,85,85,80]
#print mean(arr), np.std(arr) 

dataset = (genfromtxt('C:\Users\nwoslab2\Downloads\PDProx\CovType.csv', delimiter=','))
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
sc = MinMaxScaler(feature_range=(0,1))       
Xr = sc.fit_transform(Xr)
"""
stdsc = StandardScaler().fit(Xr)
Xr = stdsc.fit_transform(Xr)

# 70% data for training, remaining for testing
scores = []

for i in range(5):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xr, Y, test_size=0.3)
    Xtrain = np.asarray(Xtrain)
    Ytrain = np.asarray(Ytrain)
    
    #print Ytrain, Ytest 
    [n, d] = Xtrain.shape
    
    noise = 0.3
    leny = np.asarray(Ytrain).shape[0]
    for i in range(int(noise*leny)):
        r = random.randint(0,leny-1)
        if(Ytrain[r]==1):
            Ytrain[r] = -1
        else:
            Ytrain[r] = 1
    
    """
    clf = svm.SVC(C=1)    
    clf.fit(Xtrain,Ytrain)
    print clf.n_support_
    sc = clf.score(Xtest,Ytest) * 100
    print "SVM score is ",sc
    scores.append(sc)
    """
       
    #Regularization Parameter
    lam = 0.0001
    T = 1000
    wt, at = PDProxDual_train(T,n,d,Xtrain,Ytrain,lam) 
    #print PDProxDual_SupportVectorRatio(wt,Xtrain,Ytrain)
    Acc = PDProxDual_predict(wt,Xtest, Ytest)
    print ("PDProx Acc is ",Acc)
    scores.append(Acc)
    

print ("Accuracy averaged over 10 runs is ",mean(scores)," and standard dev is ", np.std(scores))

"""


# THE FOLLOWING CODE IS TO PERFORM 10-FOLD CROSS VALIDATION TO FIND THE OPTIMAL PARAMETER
   
[n, d] = Xr.shape

modelW = np.zeros(d)
Acc = 0

kf = KFold(n_splits=10)

for train_index, test_index in kf.split(Xr):
    print "new fold"
    #print "training set indices are ",train_index," Testing set indices are ", test_index
    X, Xval = [Xr[i] for i in train_index],  [Xr[i] for i in test_index]
    y, Yval =  [Y[i] for i in train_index],  [Y[i] for i in test_index]
    X = np.asarray(X)
    y = np.asarray(y)
    Xval = np.asarray(Xval)
    Yval = np.asarray(Yval)
    
    noise = 0
    leny = np.asarray(y).shape[0]
    for i in range(int(noise*leny)):
        r = random.randint(0,leny-1)
        if(y[r]==1):
            y[r] = -1
        else:
            y[r] = 1

    lenYval = np.asarray(Yval).shape[0]       
    for i in range(int(noise*lenYval)):
        r = random.randint(0,lenYval-1)
        if(Yval[r]==1):
            Yval[r] = -1
        else:
            Yval[r] = 1
    
    [n, d] = X.shape
    bestw = np.zeros(d)
    maxAcc = 0
    lam = 0.001
    T = 1000
    
    for i in range(5) :
        print "for lambda ",lam
        wt, at = PDProxDual_train(T,n,d,X,y,lam) 
        Acc = PDProxDual_predict(wt,Xval, Yval)
        print lam,Acc
        if(Acc > maxAcc):
            bestw = wt
            maxAcc = Acc
        lam = lam + 0.01
    
    
    modelW = modelW + bestw
    #modelB = modelB + b
    #print modelB.shape, b.shape
"""
