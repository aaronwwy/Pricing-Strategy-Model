'''
This script is feature selection which use following method for ADS model development:
1. xgboost selection
2. random forest selection
3. high correlation remove
Author: Wangyang Wu
Date: 09/10/2017
'''

import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt 
import seaborn as sns 

def select_xgboost(df,target,imp_num=None):
    x = df.drop([target],axis=1)
    y = df[target]
    xgmodel = XGBClassifier(random_state=0)
    xgmodel = xgmodel.fit(x,y,eval_metric='auc')
    xg_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':xgmodel.feature_importances_})
    xg_fea_imp = xg_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    xg_select_col = list(xg_fea_imp.col)
    return xg_fea_imp,xg_select_col

def select_xgboost_regression(df,target,imp_num=None):
    x = df.drop([target],axis=1)
    y = df[target]
    xgmodel = XGBRegressor(random_state=0)
    xgmodel = xgmodel.fit(x,y,eval_metric='auc')
    xg_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':xgmodel.feature_importances_})
    xg_fea_imp = xg_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    xg_select_col = list(xg_fea_imp.col)
    return xg_fea_imp,xg_select_col


def select_rf(df,target,imp_num=None):
    x = df.drop([target],axis=1)
    y = df[target]
    rfmodel = RandomForestClassifier(random_state=0)
    rfmodel = rfmodel.fit(x,y)
    rf_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':rfmodel.feature_importances_})
    rf_fea_imp = rf_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    rf_select_col = list(rf_fea_imp.col)
    return rf_fea_imp,rf_select_col

def select_rf_regression(df,target,imp_num=None):
    x = df.drop([target],axis=1)
    y = df[target]
    rfmodel = RandomForestRegressor(random_state=0)
    rfmodel = rfmodel.fit(x,y)
    rf_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':rfmodel.feature_importances_})
    rf_fea_imp = rf_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    rf_select_col = list(rf_fea_imp.col)
    return rf_fea_imp,rf_select_col


def plot_corr(df,col_list,threshold=None,plt_size=None,is_annot=True):
    corr_df = df.loc[:,col_list].corr()
    plt.figure(figsize=plt_size)
    sns.heatmap(corr_df,annot=is_annot,cmap='rainbow',vmax=1,vmin=-1,mask=np.abs(corr_df)<=threshold)
    return plt.show()


def forward_delete_corr(df,col_list,threshold=None):
    list_corr = col_list[:]
    for col in list_corr:
        corr = df.loc[:,list_corr].corr()[col]
        corr_index= [x for x in corr.index if x!=col]
        corr_values  = [x for x in corr.values if x!=1]
        for i,j in zip(corr_index,corr_values):
            if abs(j)>=threshold:
                list_corr.remove(i)
    return list_corr



def corr_mapping(df,col_list,threshold=None):
    corr_df = df.loc[:,col_list].corr()
    col_a = []
    col_b = []
    corr_value = []
    for col,i in zip(col_list[:-1],range(1,len(col_list),1)):
        high_corr_col=[]
        high_corr_value=[]
        corr_series = corr_df[col][i:]
        for i,j in zip(corr_series.index,corr_series.values):
            if abs(j)>=threshold:
                high_corr_col.append(i)
                high_corr_value.append(j)
        col_a.extend([col]*len(high_corr_col))
        col_b.extend(high_corr_col)
        corr_value.extend(high_corr_value)

    corr_map_df = pd.DataFrame({'col_A':col_a,
                                'col_B':col_b,
                                'corr':corr_value})
    return corr_map_df
