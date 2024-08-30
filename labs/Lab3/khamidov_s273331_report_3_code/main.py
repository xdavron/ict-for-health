import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def findROC(x,y):# 
    """ findROC(x,y) generates data to plot the ROC curve.
    x and y are two 1D vectors each with length N
    x[k] is the scalar value measured in the test
    y[k] is either 0 (healthy person) or 1 (ill person)
    The output data is a 2D array N rows and three columns
    data[:,0] is the set of thresholds
    data[:,1] is the corresponding false alarm
    data[:,2] is the corresponding sensitivity"""
    
    if x.min()>0:# add a couple of zeros, in order to have the zero threshold
        x=np.insert(x,0,0)# add a zero as the first element of xs
        y=np.insert(y,0,0)# also add a zero in y
    
    ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
    ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
    x0=x[ii0]# test values for healthy patients
    x1=x[ii1]# test values for ill patients
    xs=np.sort(x)# sort test values: they represent all the possible  thresholds
    # if x> thresh -> test is positive
    # if x <= thresh -> test is negative
    # number of cases for which x0> thresh represent false positives
    # number of cases for which x0<= thresh represent true negatives
    # number of cases for which x1> thresh represent true positives
    # number of cases for which x1<= thresh represent false negatives
    # sensitivity = P(x>thresh|the patient is ill)=
    #             = P(x>thresh, the patient is ill)/P(the patient is ill)
    #             = number of positives in x1/number of positives in y
    # false alarm = P(x>thresh|the patient is healthy)
    #             = number of positives in x0/number of negatives in y
    Np=ii1.size# number of positive cases
    Nn=ii0.size# number of negative cases
    data=np.zeros((Np+Nn,3),dtype=float)
    i=0
    ROCarea=0
    for thresh in xs:
        n1=np.sum(x1>thresh)#true positives
        sens=n1/Np
        n2=np.sum(x0>thresh)#false positives
        falsealarm=n2/Nn
        data[i,0]=thresh
        data[i,1]=falsealarm
        data[i,2]=sens
        if i>0:
            ROCarea=ROCarea+sens*(data[i-1,1]-data[i,1])
        i=i+1
    return data,ROCarea
#%%
def find_DTp_DTn(x,y):# 
    """ find_DTp_DTn(x,y) generates data to plot the ROC curve.
    x and y are two 1D vectors each with length N
    x[k] is the scalar value measured in the test
    y[k] is either 0 (healthy person) or 1 (ill person)
    The output data is a 2D array N rows and three columns
    data[:,0] is the set of thresholds
    data[:,1] is the corresponding false alarm
    data[:,2] is the corresponding sensitivity"""
    prev = 0.02
    if x.min()>0:# add a couple of zeros, in order to have the zero threshold
        x=np.insert(x,0,0)# add a zero as the first element of xs
        y=np.insert(y,0,0)# also add a zero in y
    
    ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
    ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
    x0=x[ii0]# test values for healthy patients
    x1=x[ii1]# test values for ill patients
    xs=np.sort(x)# sort test values: they represent all the possible  thresholds
    # if x> thresh -> test is positive
    # if x <= thresh -> test is negative
    # number of cases for which x0> thresh represent false positives
    # number of cases for which x0<= thresh represent true negatives
    # number of cases for which x1> thresh represent true positives
    # number of cases for which x1<= thresh represent false negatives
    # sensitivity = P(x>thresh|the patient is ill)=
    #             = P(x>thresh, the patient is ill)/P(the patient is ill)
    #             = number of positives in x1/number of positives in y
    # false alarm = P(x>thresh|the patient is healthy)
    #             = number of positives in x0/number of negatives in y
    Np=ii1.size# number of positive cases
    Nn=ii0.size# number of negative cases
    data=np.zeros((Np+Nn,3),dtype=float)
    i=0
    for thresh in xs:
        n1=np.sum(x1>thresh)#true positives
        sens=n1/Np
        n2=np.sum(x0>thresh)#false positives
        falsealarm=n2/Nn
        Tp=sens*prev+falsealarm*(1-prev)
        D_Tp=sens*prev/Tp
        Tn=(1-sens)*prev+(1-falsealarm)*(1-prev)
        D_Tn=(1-sens)*prev/Tn
        data[i,0]=thresh
        data[i,1]=D_Tp
        data[i,2]=D_Tn
        i=i+1
    return data
#%%
def find_HTp_HTn(x,y):# 
    """ find_DTp_DTn(x,y) generates data to plot the ROC curve.
    x and y are two 1D vectors each with length N
    x[k] is the scalar value measured in the test
    y[k] is either 0 (healthy person) or 1 (ill person)
    The output data is a 2D array N rows and three columns
    data[:,0] is the set of thresholds
    data[:,1] is the corresponding false alarm
    data[:,2] is the corresponding sensitivity"""
    prev = 0.02
    if x.min()>0:# add a couple of zeros, in order to have the zero threshold
        x=np.insert(x,0,0)# add a zero as the first element of xs
        y=np.insert(y,0,0)# also add a zero in y
    
    ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
    ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
    x0=x[ii0]# test values for healthy patients
    x1=x[ii1]# test values for ill patients
    xs=np.sort(x)# sort test values: they represent all the possible  thresholds
    # if x> thresh -> test is positive
    # if x <= thresh -> test is negative
    # number of cases for which x0> thresh represent false positives
    # number of cases for which x0<= thresh represent true negatives
    # number of cases for which x1> thresh represent true positives
    # number of cases for which x1<= thresh represent false negatives
    # sensitivity = P(x>thresh|the patient is ill)=
    #             = P(x>thresh, the patient is ill)/P(the patient is ill)
    #             = number of positives in x1/number of positives in y
    # false alarm = P(x>thresh|the patient is healthy)
    #             = number of positives in x0/number of negatives in y
    Np=ii1.size# number of positive cases
    Nn=ii0.size# number of negative cases
    data=np.zeros((Np+Nn,3),dtype=float)
    i=0
    for thresh in xs:
        n1=np.sum(x1>thresh)#true positives
        sens=n1/Np
        n2=np.sum(x0>thresh)#false positives
        falsealarm=n2/Nn
        Tp=sens*prev+falsealarm*(1-prev)
        H_Tp=falsealarm*(1-prev)/Tp
        Tn=(1-sens)*prev+(1-falsealarm)*(1-prev)
        H_Tn=(1-falsealarm)*(1-prev)/Tn
        data[i,0]=thresh
        data[i,1]=H_Tp
        data[i,2]=H_Tn
        i=i+1
    return data
#%%
plt.close('all')
xx=pd.read_csv("data/covid_serological_results.csv")
swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1 = unclear, 2=illness
Test1=xx.IgG_Test1_titre.values
Test2=xx.IgG_Test2_titre.values
ii=np.argwhere(swab==1).flatten()
swab=np.delete(swab,ii)
swab=swab//2
Test1=np.delete(Test1,ii)
Test2=np.delete(Test2,ii)
#%%DBSCAN
#Test2re = Test2.reshape(-1,1)
dbscan1 = DBSCAN(eps=2.8, min_samples=13).fit(Test1.reshape(-1,1))
labels1 = dbscan1.labels_
n_clusters1_ = len(set(labels1)) - (1 if -1 in labels1 else 0)
n_noise1_ = list(labels1).count(-1)
count_0 = list(labels1).count(0)
count_1 = list(labels1).count(1)

count_true_0 = list(swab).count(0)
count_true_1 = list(swab).count(1)
c1=0 
c2=0

outlier1=np.argwhere(labels1==-1).flatten()
Test1_no_out=np.delete(Test1,outlier1)
swab_no_out=np.delete(swab,outlier1)
for i,item in enumerate(swab):
    if i in outlier1:
        if item == 0:
            c1+=1
        else:
            c2+=1
print(f'{c1} {c2}')
#%%#%% 
ii0=np.argwhere(swab==0)
ii1=np.argwhere(swab==1)
plt.figure()
plt.hist(Test2[ii0],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test2[ii1],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()
plt.text(0., 0., 'ICT for Health', 
         fontsize=40, color='gray', alpha=0.5,
        ha='left', va='bottom', rotation='30')
plt.title('Test2')

# ii0_1_old=np.argwhere(swab==0)
# ii1_1_old=np.argwhere(swab==1)
# plt.figure()
# plt.hist(Test1[ii0_1_old],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
# plt.hist(Test1[ii1_1_old],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
# plt.grid()
# plt.legend()
# plt.text(0., 0., 'ICT for Health', 
#          fontsize=40, color='gray', alpha=0.5,
#         ha='left', va='bottom', rotation='30')
# plt.title('Test1 old')

ii0_1=np.argwhere(swab_no_out==0)
ii1_1=np.argwhere(swab_no_out==1)
plt.figure()
plt.hist(Test1_no_out[ii0_1],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
plt.hist(Test1_no_out[ii1_1],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
plt.grid()
plt.legend()
plt.text(0., 0., 'ICT for Health', 
         fontsize=40, color='gray', alpha=0.5,
        ha='left', va='bottom', rotation='30')
plt.title('Test1')
#%%   
data_Test2,area=findROC(Test2,swab)

plt.figure()
plt.plot(data_Test2[:,1],data_Test2[:,2],'-',label='Test2')
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend()
plt.title('ROC - ')

plt.figure()
plt.plot(data_Test2[:,0],data_Test2[:,1],'.',label='False alarm')
plt.plot(data_Test2[:,0],data_Test2[:,2],'.',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()

plt.figure()
plt.plot(data_Test2[:,0],1-data_Test2[:,1],'-',label='Specificity')
plt.plot(data_Test2[:,0],data_Test2[:,2],'-',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()
#%%
data_Test1,area1=findROC(Test1_no_out,swab_no_out)

plt.figure()
plt.plot(data_Test1[:,1],data_Test1[:,2],'-',label='Test1')
plt.xlabel('FA')
plt.ylabel('Sens')
plt.grid()
plt.legend(loc='lower right')
plt.title('ROC - ')

plt.figure()
plt.plot(data_Test1[:,0],data_Test1[:,1],'.',label='False alarm')
plt.plot(data_Test1[:,0],data_Test1[:,2],'.',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1')
plt.grid()

plt.figure()
plt.plot(data_Test1[:,0],1-data_Test1[:,1],'-',label='Specificity')
plt.plot(data_Test1[:,0],data_Test1[:,2],'-',label='Sensitivity')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1')
plt.grid()
#%%
data_Test2_Dtp_Dtn=find_DTp_DTn(Test2,swab)

plt.figure()
plt.plot(data_Test2_Dtp_Dtn[:,0],data_Test2_Dtp_Dtn[:,1],label='P(D|Tp)')
plt.plot(data_Test2_Dtp_Dtn[:,0],data_Test2_Dtp_Dtn[:,2],label='P(D|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()
#%%
data_Test1_Dtp_Dtn=find_DTp_DTn(Test1_no_out,swab_no_out)

plt.figure()
plt.plot(data_Test1_Dtp_Dtn[:,0],data_Test1_Dtp_Dtn[:,1],label='P(D|Tp)')
plt.plot(data_Test1_Dtp_Dtn[:,0],data_Test1_Dtp_Dtn[:,2],label='P(D|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1')
plt.grid()
#%%
data_Test2_Htp_Htn=find_HTp_HTn(Test2,swab)

plt.figure()
plt.plot(data_Test2_Htp_Htn[:,0],data_Test2_Htp_Htn[:,1],label='P(H|Tp)')
plt.plot(data_Test2_Htp_Htn[:,0],data_Test2_Htp_Htn[:,2],label='P(H|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test2')
plt.grid()
#%%
data_Test1_Htp_Htn=find_HTp_HTn(Test1_no_out,swab_no_out)

plt.figure()
plt.plot(data_Test1_Htp_Htn[:,0],data_Test1_Htp_Htn[:,1],label='P(H|Tp)')
plt.plot(data_Test1_Htp_Htn[:,0],data_Test1_Htp_Htn[:,2],label='P(H|Tn)')
plt.legend()
plt.xlabel('threshold')
plt.title('Test1')
plt.grid()