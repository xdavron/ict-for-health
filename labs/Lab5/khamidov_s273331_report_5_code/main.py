import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import graphviz 


def shuffle_data(data, seed):
    Np, Nc = data.shape
    indexsh = np.arange(Np)
    np.random.seed(seed)  # set the seed for random shuffling
    np.random.shuffle(indexsh)
    shuffled_data = data.copy(deep=True)
    shuffled_data = shuffled_data.set_axis(indexsh, axis=0, inplace=False)
    shuffled_data = shuffled_data.sort_index(axis=0)
    return shuffled_data


def run_decision_tree(train,test,n=0):
    target_names = ['notckd','ckd']
    labels = train.loc[:,'classk']
    data = train.drop('classk', axis=1)
    clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
    clfXtrain = clfXtrain.fit(data,labels)
    test_pred = clfXtrain.predict(test.drop('classk', axis=1))
    from sklearn.metrics import accuracy_score
    print('accuracy =', accuracy_score(test.loc[:,'classk'],test_pred))
    from sklearn.metrics import confusion_matrix
    print('Confusion matrix')
    print(confusion_matrix(test.loc[:,'classk'],test_pred))
    dot_data = tree.export_graphviz(clfXtrain, out_file=None,feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render("Tree_Xtrain {}".format(n)) 
    
    
def run_random_forest(train,test):
    labels = train.loc[:,'classk']
    data = train.drop('classk', axis=1)
    RF = RandomForestClassifier(n_estimators=10, random_state=0)
    RF = RF.fit(data,labels)
    test_pred = RF.predict(test.drop('classk', axis=1))
    from sklearn.metrics import accuracy_score
    print('accuracy =', accuracy_score(test.loc[:,'classk'],test_pred))
    from sklearn.metrics import confusion_matrix
    print('Confusion matrix')
    print(confusion_matrix(test.loc[:,'classk'],test_pred))
#%%
# define the feature names:
feat_names=['age','bp','sg','al','su','rbc','pc',
'pcc','ba','bgr','bu','sc','sod','pot','hemo',
'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
'ane','classk']
ff=np.array(feat_names)
feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
         'num','num','num','num','num','num','num','num','num',
         'cat','cat','cat','cat','cat','cat','cat'])
# import the dataframe:
#xx=pd.read_csv("./data/chronic_kidney_disease.arff",sep=',',
#               skiprows=29,names=feat_names, 
#               header=None,na_values=['?','\t?'],
#               warn_bad_lines=True)
xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
    skiprows=29,names=feat_names, 
    header=None,na_values=['?','\t?'],)
Np,Nf=xx.shape
#%% change categorical data into numbers:
key_list=["normal","abnormal","present","notpresent","yes",
"no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
xx=xx.replace(key_list,key_val)
print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values

#%% manage the missing data through regression
print(xx.info())
x=xx.copy()
# drop rows with less than 19=Nf-6 recorded features:
x=x.dropna(thresh=19)
x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
n=x.isnull().sum(axis=1)# check the number of missing values in each row
print('max number of missing values in the reduced dataset: ',n.max())
print('number of points in the reduced dataset: ',len(n))
# take the rows with exctly Nf=25 useful features; this is going to be the training dataset
# for regression
Xtrain=x.dropna(thresh=25)
Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
# get the possible values (i.e. alphabet) for the categorical features
alphabets=[]
for k in range(len(feat_cat)):
    if feat_cat[k]=='cat':
        val=Xtrain.iloc[:,k]
        val=val.unique()
        alphabets.append(val)
    else:
        alphabets.append('num')

#%% run regression tree on all the missing data
#normalize the training dataset
mm=Xtrain.mean(axis=0)
ss=Xtrain.std(axis=0)
Xtrain_norm=(Xtrain-mm)/ss
# get the data subset that contains missing values 
Xtest=x.drop(x[x.isnull().sum(axis=1)==0].index)
Xtest.reset_index(drop=True, inplace=True)# reset the index of the dataframe
Xtest_norm=(Xtest-mm)/ss # nomralization
Np,Nf=Xtest_norm.shape
regr=tree.DecisionTreeRegressor() # instantiate the regressor
for kk in range(Np):
    xrow=Xtest_norm.iloc[kk]#k-th row
    mask=xrow.isna()# columns with nan in row kk
    Data_tr_norm=Xtrain_norm.loc[:,~mask]# remove the columns from the training dataset
    y_tr_norm=Xtrain_norm.loc[:,mask]# columns to be regressed
    regr=regr.fit(Data_tr_norm,y_tr_norm)
    Data_te_norm=Xtest_norm.loc[kk,~mask].values.reshape(1,-1) # row vector
    ytest_norm=regr.predict(Data_te_norm)
    Xtest_norm.iloc[kk][mask]=ytest_norm # substitute nan with regressed values
Xtest_new=Xtest_norm*ss+mm # denormalize
# substitute regressed numerical values with the closest element in the alphabet
index=np.argwhere(feat_cat=='cat').flatten()
for k in index:
    val=alphabets[k].flatten() # possible values for the feature
    c=Xtest_new.iloc[:,k].values # values in the column
    c=c.reshape(-1,1)# column vector
    val=val.reshape(1,-1) # row vector
    d=(val-c)**2 # matrix with all the distances w.r.t. the alphabet values
    ii=d.argmin(axis=1) # find the index of the closest alphabet value
    Xtest_new.iloc[:,k]=val[0,ii]
print(Xtest_new.nunique())
print(Xtest_new.describe().T)
#
X_new= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
##------------------ Decision tree -------------------
## first decision tree, using Xtrain for training and Xtest_new for test
target_names = ['notckd','ckd']
labels = Xtrain.loc[:,'classk']
data = Xtrain.drop('classk', axis=1)
clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
clfXtrain = clfXtrain.fit(data,labels)
test_pred = clfXtrain.predict(Xtest_new.drop('classk', axis=1))
from sklearn.metrics import accuracy_score
print('accuracy =', accuracy_score(Xtest_new.loc[:,'classk'],test_pred))
from sklearn.metrics import confusion_matrix
print('Confusion matrix')
print(confusion_matrix(Xtest_new.loc[:,'classk'],test_pred))
#%% export to graghviz to draw a grahp
dot_data = tree.export_graphviz(clfXtrain, out_file=None,feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("Tree_Xtrain") 
# %% Generate the shuffled data
X_new_sh = shuffle_data(X_new, 273331)
# Generate training and testing data
X_tr_1 = X_new_sh[0:158]
X_te_1 = X_new_sh[159:]
run_decision_tree(X_tr_1, X_te_1, 1)
# %% Generate the shuffled data
X_new_sh_2 = shuffle_data(X_new, 273332)
# Generate training and testing data
X_tr = X_new_sh_2[0:158]
X_te = X_new_sh_2[159:]
run_decision_tree(X_tr, X_te, 2)
# %% Generate the shuffled data
X_new_sh_3 = shuffle_data(X_new, 273333)
# Generate training and testing data
X_tr = X_new_sh_3[0:158]
X_te = X_new_sh_3[159:]
run_decision_tree(X_tr, X_te, 3)
# %% Random Forest
run_random_forest(X_tr_1,X_te_1)


