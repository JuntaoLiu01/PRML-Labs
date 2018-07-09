from __future__ import division
import os,math
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix,make_scorer,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib

from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalanceCascade 
from collections import Counter

FEAT_DICT = {'vhigh':4,'high':3,'med':2,'low':1,
             '2':2,'3':3,'4':4,'5more':5,
             'more':5,
             'small':1,'big':3,
             }
LABEL_DICT = {'positive':1,'negative':0}
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def encode_feature(l,data_type=1):
    ret = []
    for l_ in l:
        if data_type == 1:
            l_ = FEAT_DICT[l_]
        else:
            l_ = float(l_)
        ret.append(l_)
    return ret

def load_data(filename,data_type=1):
    data = []
    with open(filename,'r') as rf:
        for line in rf:
            if line[0] != '@':
                line = line.replace(' ','')
                item = line.strip().split(',')
                item[:-1] = encode_feature(item[:-1],data_type)
                item[-1] = LABEL_DICT[item[-1]]
                data.append(item)
        return np.array(data)

def one_hot_encode_feature(feat):
    enc = OneHotEncoder()
    feat = enc.fit_transform(feat).toarray()
    return feat

def train_SVC(X,y):
    model = SVC(kernel='rbf')
    model.fit(X,y)
    joblib.dump(model,'model.m')
    return model

def eval_model(model,X_test,y_test):
    labels = [1,0]
    target_names = ['postive','negative']
    report = classification_report(y_test,model.predict(X_test),labels=labels,target_names=target_names)
    print(report)

def TP(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)[1,1]
def TN(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)[0,0]
def FP(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)[0,1]
def FN(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)[1,0]
def G_mean(y_true,y_pred):
    matrix = confusion_matrix(y_true,y_pred)
    pos_acc = matrix[1,1]/(matrix[1,1] + matrix[1,0])
    neg_acc = matrix[0,0]/(matrix[0,0] + matrix[0,1])
    return math.sqrt(pos_acc*neg_acc)

def cross_val(feat,label,gamma=None):
    if gamma is not None:
        model = SVC(kernel='rbf',gamma=gamma,class_weight='balanced')
    else:
        model = LogisticRegression()
    socring = { 
                '1_TP':make_scorer(TP),
                '2_TN':make_scorer(TN),
                '3_FP':make_scorer(FP),
                '4_FN':make_scorer(FN),
                '5_F1':'f1',
                '6_AUC':'roc_auc',
                '7_G-mean':make_scorer(G_mean)
              }
    scores = cross_validate(model,feat,label,scoring=socring,cv=5,return_train_score=False)
    ret = {}
    for key,value in scores.items():
            if key[0] == 't':
                ret[key[5:]]=value[-1]
    return ret

def multi_cross_val(feat,label,gamma=None,times=1):
    if gamma is not None:
        results = {'0_gamma':[gamma]}
    else:
        results = {}
    for i in range(times):
        ret = cross_val(feat,label,gamma)
        for key,value in ret.items():
            if key in results:
                results[key].append(value)
            else:
                results[key] = [value]
    for key,value in results.items():
        results[key] = np.array(value).mean()
    results = sorted(results.items(),key=lambda item:item[0])
    return results

def save_results(results,filename):
    with open(filename,'a') as wf:
        for item in results:
            wf.write(str(item[1]))
            wf.write('\t')
        wf.write('\n')

def baseline(filename,savename,data_type):
    data = load_data(filename,data_type)
    data_feat = data[:,:-1];data_label = data[:,-1]
    if data_type == 1:
        data_feat = one_hot_encode_feature(data_feat)
    results = multi_cross_val(data_feat,data_label)
    save_results(results,savename)
    # X_train,X_test,y_train,y_test = train_test_split(data_feat,data_label,test_size=0.2,random_state=19)
    # model = train_SVC(X_train,y_train)
    # eval_model(model,X_test,y_test)

def borderline_smote(filename,savename,data_type):
    data = load_data(filename,data_type)
    data_feat = data[:,:-1];data_label = data[:,-1]
    if data_type == 1:
        data_feat = one_hot_encode_feature(data_feat)
    print(Counter(data_label))
    sm = SMOTE(random_state=19,kind='borderline1',k_neighbors=6)
    feat_res,label_res = sm.fit_sample(data_feat,data_label)
    print(Counter(label_res))
    results = multi_cross_val(feat_res,label_res)
    save_results(results,savename)

def ensemble_adaboost(feat,label):
    print(type(label))
    print(Counter(label))
    bm = BalanceCascade(random_state=19,estimator='adaboost')
    feat_res,label_res = bm.fit_sample(feat,label)
    print(label_res.shape)
    return feat_res,label_res

if __name__ == '__main__':
    filename = os.path.join(BASE_PATH,'dataset/yeast/yeast-2_vs_4.dat')
    savename = os.path.join(BASE_PATH,'dataset/yeast/results.txt')
    borderline_smote(filename,savename,2)
    # data = load_data(filename,3)
    # data_feat = data[:,:-1]
    # data_label = data[:,-1].astype('int')
    # # data_feat = one_hot_encode_feature(data_feat)
    # data_feat,data_label = ensemble_adaboost(data_feat,data_label)
    