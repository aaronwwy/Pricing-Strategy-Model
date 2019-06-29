'''
This script is common functions which is used for ADS model development
Author: Wangyang Wu
Date: 08/26/2017
'''

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn import ensemble,tree,linear_model,preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,RandomForestRegressor,AdaBoostRegressor
from sklearn.feature_selection import chi2,SelectKBest,RFECV

from sklearn.metrics import r2_score,accuracy_score,mean_squared_error,log_loss,roc_curve, auc,log_loss,precision_recall_curve,average_precision_score
from sklearn.preprocessing import StandardScaler,scale
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
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

def summarize_dataset(df, Class):
    cols1 = df.columns
    # shape
    print(df[cols1].shape)
    # head
    print(df[cols1].head(5))
    # descriptions
    print(df[cols1].describe())
    # class distribution
    print(df.groupby(Class).size())

def describe_table(DF):
    df=DF.copy()
    df.reset_index(inplace=True)
    df_input_dtypes = pd.DataFrame(df.dtypes,columns=['Variable Type'])
    df_input_dtypes = df_input_dtypes.reset_index()
    df_input_dtypes['Variable Name'] = df_input_dtypes['index']
    df_input_dtypes = df_input_dtypes[['Variable Name','Variable Type']]
    #df_input_dtypes['Sample Value'] = df.loc[0].values
    
    unique_val=pd.DataFrame(df.nunique()).reset_index()
    unique_val.rename(columns={'index':'Variable Name',
                               0:'Unique Values'},
    inplace=True)
    
    mis_val = df.isnull().sum()
    mis_val_percent = df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    #mis_val_table[2]=map(lambda x: "{0:.2f}%".format(x * 100),mis_val_table[1])
    mis_val_table[2]=mis_val_table[1].apply(lambda x: "{0:.2f}%".format(x * 100))
    mis_val_table[1]=mis_val_table[1].apply(lambda x: round(float(x),2))
    #mis_val_table[1]=map(lambda x: round(float(x),2),mis_val_table[1])
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Number of Missing Value', 
               1 : 'Float of Missing Value',
               2 : 'Percentage of Missing Value'}
               )
    mis_val_table_ren_columns = mis_val_table_ren_columns.reset_index()
    mis_val_table_ren_columns['Variable Name'] = mis_val_table_ren_columns['index']
    mis_val_table_ren_columns = mis_val_table_ren_columns[['Variable Name','Number of Missing Value', 'Float of Missing Value','Percentage of Missing Value']]
	 
    df_input_dtypes=df_input_dtypes.merge(unique_val,left_on='Variable Name', right_on='Variable Name',how='left')
    df_input_dtypes=df_input_dtypes.merge(mis_val_table_ren_columns, left_on='Variable Name', right_on='Variable Name',how='left')
    
    df_input_dtypes['Variable Type']=df_input_dtypes['Variable Type'].astype(str)
    df_input_dtypes.sort_values(['Number of Missing Value'],ascending=False,inplace=True)
    df_input_dtypes.reset_index(inplace=True,drop=True)
        
    return df_input_dtypes.drop(['Number of Missing Value','Float of Missing Value'],1)

def missing_cal(df):
    missing_series = df.isnull().sum()/df.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index':'col',
                                            0:'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct',ascending=False).reset_index(drop=True)
    return missing_df


def plot_missing_var(df,plt_size=None):
    missing_df = missing_cal(df)
    plt.figure(figsize=plt_size)
    plt.rcParams['axes.unicode_minus'] = False
    x = missing_df['missing_pct']
    plt.hist(x=x,bins=np.arange(0,1.1,0.1),color='hotpink',ec='k',alpha=0.8)
    plt.ylabel('Number of Missing')
    plt.xlabel('Percentage of Missing')
    return plt.show()



def plot_missing_user(df,plt_size=None):
    missing_series = df.isnull().sum(axis=1)
    list_missing_num  = sorted(list(missing_series.values))
    plt.figure(figsize=plt_size)
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(df.shape[0]),list_missing_num)
    plt.ylabel('Number of Variables have Missing Value')
    plt.xlabel('sanples')
    return plt.show()



def missing_delete_var(df,threshold=None):
    df2 = df.copy()
    missing_df = missing_cal(df)
    missing_col_num = missing_df[missing_df.missing_pct>=threshold].shape[0]
    missing_col = list(missing_df[missing_df.missing_pct>=threshold].col)
    df2 = df2.drop(missing_col,axis=1)
    print('There are {} variables that have percentage of missing more than {}'.format(missing_col_num, threshold))
    return df2


def fillna_cate_var(df,col_list,fill_type=None):
    df2 = df.copy()
    for col in col_list:
        if fill_type=='class':
            df2[col] = df2[col].fillna('unknown')
        if fill_type=='mode':
            df2[col] = df2[col].fillna(df2[col].mode()[0])
    return df2


def fillna_num_var(df,col_list,fill_type=None,filled_df=None):
    df2 = df.copy()
    for col in col_list:
        if fill_type=='median':
            df2[col] = df2[col].fillna(df2[col].median())
        if fill_type=='class':
            df2[col] = df2[col].fillna(-999)
        if fill_type=='rf':
            rf_df = pd.concat([df2[col],filled_df],axis=1)
            known = rf_df[rf_df[col].notnull()]
            unknown = rf_df[rf_df[col].isnull()]
            x_train = known.drop([col],axis=1)
            y_train = known[col]
            x_pre = unknown.drop([col],axis=1)
            rf = RandomForestRegressor(random_state=0)
            rf.fit(x_train,y_train)
            y_pre = rf.predict(x_pre)
            df2.loc[df2[col].isnull(),col] = y_pre
    return df2


def const_delete(df,col_list,threshold=None):

    df2 = df.copy()
    const_col = []
    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0]/df2[df2[col].notnull()].shape[0]
        if const_pct>=threshold:
            const_col.append(col)
    df2 = df2.drop(const_col,axis=1)
    print('Number of Variables that only have single value {}'.format(len(const_col)))
    return df2



def descending_cate(df,col_list,threshold=None):
    df2 = df.copy()
    for col in col_list:
        value_series = df[col].value_counts()/df[df[col].notnull()].shape[0]
        small_value = []
        for value_name,value_pct in zip(value_series.index,value_series.values):
            if value_pct<=threshold:
                small_value.append(value_name)
        df2.loc[df2[col].isin(small_value),col]='other'
    return df2
    

def campaign_decile(new_leads,datasets,name,score_name):        
    temp=datasets.copy()
    temp[name]=np.NAN
    temp.sort_values([score_name],ascending=False,inplace=True)
    temp.reset_index(inplace=True)
    for decile in range(0,10):
        if decile==0:
            temp[name][temp.index<=new_leads[decile]-1]=decile
        else:
            temp[name][(temp.index>sum(new_leads[:decile])-1) & (temp.index<=sum(new_leads[:decile+1])-1)]=decile
    temp[name]=temp[name]+1
    return temp
 
      
def chk_corr(dataframe,columns, threshold=0.80):
    corr_temp=dataframe[columns].corr()
    result=pd.DataFrame(columns=['Var1','Var2','Correlation'])
    i=0
    checked=[]
    for index in corr_temp.index:
        checked.append(index)
        for column in corr_temp.columns:
            if column not in checked:
                correlation=corr_temp.loc[index,column]
                if (abs(correlation)>=threshold and correlation<1.0) or correlation==-1.0 :                    
                    result.loc[i,'Var1'] = index
                    result.loc[i,'Var2'] = column
                    result.loc[i,'Correlation'] = correlation
                    i+=1
    return result
                    

def two_overlap(df,decile1,decile2,customerid=None):   
    if customerid is not None:    
        leads=df.groupby([decile1,decile2])[customerid].nunique().to_frame().reset_index()
    else:
        leads=df.groupby([decile1,decile2]).size().to_frame().reset_index()
    leads.fillna(0,inplace=True)
    
    length=df[decile1].nunique()
    '''Index is Decile1, Column is Decile2'''
    leads_df=pd.DataFrame(index=sorted(df[decile1].unique()),columns=sorted(df[decile2].unique()))

    for rs in leads[decile1].values:
        for pp in leads[decile2].values:
            value=leads[(leads[decile1]==rs) & (leads[decile2]==pp)][0].values.astype(int)
            leads_df.set_value(rs,pp,value)
    print ('Column is %s and Row is %s'%(decile2,decile1)) 
    return leads_df

                   
def outliers_z_score(ys):
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

       
def outliers_modified_z_score(value,threshold):
    ys=pd.DataFrame(list(value),columns=['value'])
    median_y = ys['value'].median()
    ys['MAD']=map(lambda x:np.abs(x - median_y), ys['value'])
    median_absolute_deviation_y=ys['MAD'].median()
    if median_absolute_deviation_y>0:
        ys['mz_score']=map(lambda x: 0.6745 * (x - median_y) / median_absolute_deviation_y,ys['value'])
        correct=ys['value'][np.abs(ys['mz_score'])<=threshold]
        correct_min=correct.min()
        correct_max=correct.max()
        num_outliers=ys.shape[0]-correct.shape[0]
    
        ys['value'][ys['mz_score']<-threshold]=correct_min
        ys['value'][ys['mz_score']> threshold]=correct_max
        return ys['value'].values, int(num_outliers)
    else:
        return value, 0
        
        
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        