# -*- coding: utf-8 -*-
"""
PEP 8 -- Style Guide for Python Code
https://www.python.org/dev/peps/pep-0008/

@author: visintin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def LLS(X,y):
    w_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    sol = w_hat
    # minn = np.linalg.norm(A @ w_hat - y) ** 2
    # err = 0
    return sol


def GPR(X_train,y_train,X_val,r2,s2):
    """ Estimates the output y_val given the input X_val, using the training data 
    and  hyperparameters r2 and s2"""
    Nva=X_val.shape[0]
    yhat_val=np.zeros((Nva,))
    sigmahat_val=np.zeros((Nva,))
    for k in range(Nva):
        x=X_val[k,:]# k-th point in the validation dataset
        A=X_train-np.ones((Ntr,1))*x
        dist2=np.sum(A**2,axis=1)
        ii=np.argsort(dist2)
        ii=ii[0:N-1];
        refX=X_train[ii,:]
        Z=np.vstack((refX,x))
        sc=np.dot(Z,Z.T)# dot products
        e=np.diagonal(sc).reshape(N,1)# square norms
        D=e+e.T-2*sc# matrix with the square distances 
        R_N=np.exp(-D/2/r2)+s2*np.identity(N)#covariance matrix
        R_Nm1=R_N[0:N-1,0:N-1]#(N-1)x(N-1) submatrix 
        K=R_N[0:N-1,N-1]# (N-1)x1 column
        d=R_N[N-1,N-1]# scalar value
        C=np.linalg.inv(R_Nm1)
        refY=y_train[ii]
        mu=K.T@C@refY# estimation of y_val for X_val[k,:]
        sigma2=d-K.T@C@K
        sigmahat_val[k]=np.sqrt(sigma2)
        yhat_val[k]=mu        
    return yhat_val,sigmahat_val


plt.close('all')
xx=pd.read_csv("data/parkinsons_updrs.csv") # read the dataset
z=xx.describe().T # gives the statistical description of the content of each column
#xx.info()
# features=list(xx.columns)
features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
#%% scatter plots
todrop=['subject#', 'sex', 'test_time',  
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']
x1=xx.copy(deep=True)
X=x1.drop(todrop,axis=1)
#%% Generate the shuffled dataframe
np.random.seed(273331)
Xsh = X.sample(frac=1).reset_index(drop=True)
[Np,Nc]=Xsh.shape
F=Nc-1
#%% Generate training, validation and testing matrices
Ntr=int(Np*0.5)  # number of training points
Nva=int(Np*0.25) # number of validation points
Nte=Np-Ntr-Nva   # number of testing points
X_tr=Xsh[0:Ntr] # training dataset
# find mean and standard deviations for the features in the training dataset
mm=X_tr.mean()
ss=X_tr.std()
my=mm['total_UPDRS']# get mean for the regressand
sy=ss['total_UPDRS']# get std for the regressand
# normalize data
Xsh_norm=(Xsh-mm)/ss
ysh_norm=Xsh_norm['total_UPDRS']
Xsh_norm=Xsh_norm.drop('total_UPDRS',axis=1)
Xsh_norm=Xsh_norm.values
ysh_norm=ysh_norm.values
# get the training, validation, test normalized data
X_train_norm=Xsh_norm[0:Ntr]
X_val_norm=Xsh_norm[Ntr:Ntr+Nva]
X_test_norm=Xsh_norm[Ntr+Nva:]
y_train_norm=ysh_norm[0:Ntr]
y_val_norm=ysh_norm[Ntr:Ntr+Nva]
y_test_norm=ysh_norm[Ntr+Nva:]
y_train=y_train_norm*sy+my
y_val=y_val_norm*sy+my
y_test=y_test_norm*sy+my
#%% Apply Gaussian Process Regression
err = np.zeros((400, 3), dtype=float)
k=0
list_s2 = [6e-4, 8e-4, 1e-3, 3e-3, 5e-3]
N = 10
for i in list_s2:
    for j in np.arange(0.0, N, 0.2):
        r2=round(j, 2)
        s2=i
        yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2,s2)
        yhat_val=yhat_val_norm*sy+my
        err[k, 0] = s2
        err[k, 1] = r2
        err[k, 2] = round(np.mean((y_val-yhat_val)**2),3)
        k+=1
#%%find min
# sortederr = err[np.argsort(err[:, 2])]
dropzero = np.delete(err, np.where(err == 0)[0], axis=0)
min_err = dropzero[dropzero[:,2].argsort(axis=0)]
r2=min_err[0,1]
s2=min_err[0,0]
mse=min_err[0,2]
#%%prep for plot
colors = ['c', 'orange', 'crimson', 'darkviolet', 'brown', 'r', 'g','b']
color_index = 0
plt.figure()
for s in list_s2:
    xlist=[]
    ylist=[]
    for r in err:
        if s == r[0]:
            xlist.append(r[1])
            ylist.append(r[2])
    plt.plot(xlist,ylist,colors[color_index], label=s)
    color_index+=1
plt.plot(r2,mse,'bo')
plt.legend(loc="upper right")
plt.grid()
plt.xlabel('r2')
plt.ylabel('MSE')
plt.title('Optimization')
plt.savefig('./GP-Optimization.png')
v=plt.axis()
plt.show()

#%% Apply Gaussian Process Regression
# print(r2,s2)
yhat_train_norm,sigmahat_train=GPR(X_train_norm,y_train_norm,X_train_norm,r2,s2)
yhat_train=yhat_train_norm*sy+my
yhat_test_norm,sigmahat_test=GPR(X_train_norm,y_train_norm,X_test_norm,r2,s2)
yhat_test=yhat_test_norm*sy+my
yhat_val_norm,sigmahat_val=GPR(X_train_norm,y_train_norm,X_val_norm,r2,s2)
yhat_val=yhat_val_norm*sy+my
err_train=y_train-yhat_train
err_test=y_test-yhat_test
err_val=y_val-yhat_val
#%% plots
plt.figure()
plt.plot(y_test,yhat_test,'.b')
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5
plt.savefig('./GP-Gaussian Process Regression.png')
plt.show()


plt.figure()
plt.errorbar(y_test,yhat_test,yerr=3*sigmahat_test*sy,fmt='o',ms=2)
plt.plot(y_test,y_test,'r')
plt.grid()
plt.xlabel('y')
plt.ylabel('yhat')
plt.title('Gaussian Process Regression - with errorbars')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5
plt.savefig('./GP-Gaussian Process - with errorbars.png')
plt.show()

e=[err_train,err_val,err_test]
plt.figure()
plt.hist(e,bins=50,density=True,range=[-8,17], histtype='bar',label=['Train.','Val.','Test'])
plt.xlabel('error')
plt.ylabel('P(error in bin)')
plt.legend()
plt.grid()
plt.title('Error histogram')
v=plt.axis()
N1=(v[0]+v[1])*0.5
N2=(v[2]+v[3])*0.5
plt.savefig('./Error histogram.png')
plt.show()
#statistics GP
print('MSE train',round(np.mean((err_train)**2),3))
print('MSE test',round(np.mean((err_test)**2),3))
print('MSE valid',round(np.mean((err_val)**2),3))
print('Mean error train',round(np.mean(err_train),4))
print('Mean error test',round(np.mean(err_test),4))
print('Mean error valid',round(np.mean(err_val),4))
print('St dev error train',round(np.std(err_train),3))
print('St dev error test',round(np.std(err_test),3))
print('St dev error valid',round(np.std(err_val),3))
print('R^2 train',round(1-(np.mean((err_train)**2)/np.std(y_train)**2),4))
print('R^2 test',round(1-(np.mean((err_test)**2)/np.std(y_test)**2),4))
print('R^2 val',round(1-(np.mean((err_val)**2)/np.std(y_val)**2),4))
#%% LLS
w_hat = LLS(X_train_norm,y_train_norm)
y_hat_test_norm = X_test_norm @ w_hat
MSE_norm = np.mean((y_hat_test_norm - y_test_norm) ** 2)
MSE = sy ** 2 * MSE_norm
# plot the error histogram
E_tr = (y_train_norm - X_train_norm @ w_hat) * sy  # training
E_te = (y_test_norm - X_test_norm @ w_hat) * sy  # test
e = [E_tr, E_te]
#plot error histogram
plt.figure(figsize=(6, 4))
plt.hist(e, bins=50, density=True, histtype='bar',
         label=['training', 'test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS-Error histograms')
plt.tight_layout()
plt.savefig('./LLS-hist.png')
plt.show()

y_hat_test = (X_test_norm @ w_hat) * sy + my
y_test = y_test_norm * sy + my
#plot y versus y hat 
plt.figure(figsize=(6, 4))
plt.plot(y_test, y_hat_test, '.')
v = plt.axis()
plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
plt.xlabel(r'$y$')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./LLS-yhat_vs_y.png')
plt.show()
#%%statistics LLS
yhat_train_norm_LLS = X_train_norm @ w_hat
yhat_train_LLS=yhat_train_norm_LLS*sy+my
yhat_val_norm_LLS = X_val_norm @ w_hat
yhat_val_LLS=yhat_val_norm_LLS*sy+my
yhat_test_norm_LLS = X_test_norm @ w_hat
yhat_test_LLS=yhat_test_norm_LLS*sy+my

err_train_LLS=y_train-yhat_train_LLS
err_val_LLS=y_val-yhat_val_LLS
err_test_LLS=y_test-yhat_test_LLS
print('MSE train',round(np.mean((err_train_LLS)**2),3))
print('MSE test',round(np.mean((err_test_LLS)**2),3))
print('MSE valid',round(np.mean((err_val_LLS)**2),3))
print('Mean error train',round(np.mean(err_train_LLS),4))
print('Mean error test',round(np.mean(err_test_LLS),4))
print('Mean error valid',round(np.mean(err_val_LLS),4))
print('St dev error train',round(np.std(err_train_LLS),3))
print('St dev error test',round(np.std(err_test_LLS),3))
print('St dev error valid',round(np.std(err_val_LLS),3))
print('R^2 train',round(1-(np.mean((err_train_LLS)**2)/np.std(y_train)**2),4))
print('R^2 test',round(1-(np.mean((err_test_LLS)**2)/np.std(y_test)**2),4))
print('R^2 val',round(1-(np.mean((err_val_LLS)**2)/np.std(y_val)**2),4))