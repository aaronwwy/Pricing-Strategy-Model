'''
This script is classifcation metrics and cross validation for ADS model development:
Author: Wangyang Wu
Date: 10/2/2017
'''

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn import ensemble,tree,linear_model,preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor,AdaBoostRegressor
from sklearn.feature_selection import chi2,SelectKBest,RFECV
from sklearn import metrics
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error,log_loss,roc_curve, auc,log_loss,precision_recall_curve,average_precision_score
from sklearn.preprocessing import StandardScaler,scale
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.externals import joblib
from collections import OrderedDict
import gc
import random
import seaborn as sns
import string
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.stats.stats import pearsonr,spearmanr
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import r2_score
from scipy.special import boxcox, inv_boxcox
import os

# AUC 
def plot_roc(y_label,y_pred):
    tpr,fpr,threshold = metrics.roc_curve(y_label,y_pred) 
    AUC = metrics.roc_auc_score(y_label,y_pred) 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tpr,fpr,color='blue',label='AUC=%.3f'%AUC) 
    ax.plot([0,1],[0,1],'r--')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show(ax)


# KS 
def plot_model_ks(y_label,y_pred):
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)



def KS_Lift(predict_score, target,n_decile=10,new_leads=None,redecile=True):
    data=pd.DataFrame()
    data['prescore']=predict_score
    data['target']=target.values
    #unique_length=len(np.unique(predict_score))
    if redecile:
        data=campaign_decile(new_leads,data,'decile','prescore')
        
    #elif unique_length<10:            
        #data['decile'] =unique_length-1 - pd.qcut(data['prescore'], unique_length-1, labels=False)
    else:
        data['decile'] =n_decile- pd.qcut(data['prescore'], n_decile, labels=False,duplicates='drop')

    
    temp=pd.DataFrame(index=range(1,n_decile+1))
    lead_col='Model Obsverations Num'
    booked_col='Model Book Num'
    
    temp[lead_col]=data.groupby(['decile'])['target'].count()
    temp[booked_col]=data.groupby(['decile'])['target'].sum()    
    temp['non_booked']=temp[lead_col]-temp[booked_col]
    total_booked=temp[booked_col].sum()
    total_nonbooked=temp['non_booked'].sum()
    temp.set_value(1,'cumulative booked',temp.loc[1,booked_col])
    temp.set_value(1,'cumulative non_booked',temp.loc[1,'non_booked'])
    for i in range(2,11):
        temp.set_value(i,'cumulative booked',temp.loc[i-1,'cumulative booked']+temp.loc[i,booked_col])
        temp.set_value(i,'cumulative non_booked',temp.loc[i-1,'cumulative non_booked']+temp.loc[i,'non_booked'])
    temp['%cum booked']=(temp['cumulative booked'].astype(float))/total_booked
    temp['%cum non_booked']=(temp['cumulative non_booked'].astype(float))/total_nonbooked
    temp['diff']=temp['%cum booked']-temp['%cum non_booked']
    temp['booked rate']=(temp[booked_col].astype(float))/temp[lead_col]
    overall_rate=float(total_booked)/temp[lead_col].sum()
    temp['lift']=temp['booked rate']/overall_rate
      
    fpr, tpr, threshold = roc_curve(data['target'].values, data['prescore'].values)
    roc_auc = auc(fpr, tpr)
        
    roc_plot={}  
    roc_plot['fpr']=fpr
    roc_plot['tpr']=tpr
    roc_plot['auc']=roc_auc    
    
    
    performance={'KS':round(temp['diff'].max(),2),
                 'ROC':round(roc_auc,2),
                 'Top 1 Decile Lift':round(temp.loc[1,'lift'],2),
                 'Top 3 Deciles Capture':round(temp.loc[3,'%cum booked'],2),  
                 'Top 5 Deciles Capture':round(temp.loc[5,'%cum booked'],2)
    }
    return temp,performance,roc_plot


# cross validation
def cross_verify(x,y,estimators,fold,scoring='roc_auc'):
    cv_result = cross_val_score(estimator=estimators,X=x,y=y,cv=fold,n_jobs=-1,scoring=scoring)
    print('The max AUC in CV is:{}'.format(cv_result.max()))
    print('The min AUC in CV is:{}'.format(cv_result.min()))
    print('The average AUC in CV is:{}'.format(cv_result.mean()))
    plt.figure(figsize=(6,4))
    plt.title('Performance Metrics in Cross Validation')
    plt.boxplot(cv_result,patch_artist=True,showmeans=True,
            boxprops={'color':'black','facecolor':'yellow'},
            meanprops={'marker':'D','markerfacecolor':'tomato'},
            flierprops={'marker':'o','markerfacecolor':'red','color':'black'},
            medianprops={'linestyle':'--','color':'orange'})
    return plt.show()

def cross_verify_regerssion(x,y,estimators,fold, scoring='neg_mean_squared_error'):
    cv_result = cross_val_score(estimator=estimators,X=x,y=y,cv=fold,n_jobs=-1,scoring=scoring)
    print('The max RMSE in CV is:{}'.format(np.sqrt(-cv_result).max()))
    print('The min RMSE in CV is:{}'.format(np.sqrt(-cv_result).min()))
    print('The average RMSE in CV is:{}'.format(np.sqrt(-cv_result).mean()))
    plt.figure(figsize=(6,4))
    plt.title('Performance Metrics in Cross Validation')
    plt.boxplot(cv_result,patch_artist=True,showmeans=True,
            boxprops={'color':'black','facecolor':'yellow'},
            meanprops={'marker':'D','markerfacecolor':'tomato'},
            flierprops={'marker':'o','markerfacecolor':'red','color':'black'},
            medianprops={'linestyle':'--','color':'orange'})
    return plt.show()



def plot_learning_curve(estimator,x,y,cv=None,train_size = np.linspace(0.1,1.0,5),plt_size =None):
    from sklearn.model_selection import learning_curve
    train_sizes,train_scores,test_scores = learning_curve(estimator=estimator,
                                                          X=x,
                                                          y=y,
                                                          cv=cv,
                                                          n_jobs=-1,
                                                          train_sizes=train_size)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.figure(figsize=plt_size)
    plt.xlabel('Training-example')
    plt.ylabel('score')
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training-score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross-val-score')
    plt.legend(loc='best')
    return plt.show()


# confusion matrix
def plot_matrix_report(y_label,y_pred): 
    matrix_array = metrics.confusion_matrix(y_label,y_pred)
    plt.matshow(matrix_array, cmap=plt.cm.summer_r)
    plt.colorbar()

    for x in range(len(matrix_array)): 
        for y in range(len(matrix_array)):
            plt.annotate(matrix_array[x,y], xy =(x,y), ha='center',va='center')

    plt.xlabel('True label')
    plt.ylabel('Predict label')
    print(metrics.classification_report(y_label,y_pred))
    return plt.show()
