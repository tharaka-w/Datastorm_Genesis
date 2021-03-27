# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 07:54:54 2021

@author: Genesis
"""

import pandas as pd # import library
import numpy as np # import lib
import matplotlib.pyplot as plt

dfpolicy=pd.read_csv(r"E:\Academic\Data storm/policy_data.csv" )
#dfagent=pd.read_csv(r"E:\Academic\Data storm//Agent.csv" )

missing_values_count = dfpolicy.isnull().sum()
#missing_values_count = dfpolicy.isnull().sum()

dfpolicy['commencement_dt'] = pd.to_datetime(dfpolicy['commencement_dt'],infer_datetime_format=True)
dfpolicy["rundate2"]='01/07/2020'
dfpolicy['rundate2'] = pd.to_datetime(dfpolicy['rundate2'],infer_datetime_format=True)

dfpolicy['Duration'] = dfpolicy['rundate2']-dfpolicy['commencement_dt']
dfpolicy['Duration'] = dfpolicy.Duration.dt.days

dfpolicy = dfpolicy.drop(['commencement_dt','rundate2','termination_dt','client_code','payment_method','run_date','next_due_dt','termination_reason','policy_status','main_holder_occupation_cd','main_holder_occupation','spouse_dob','main_holder_dob','policy_code','product_name','child1_dob','child2_dob','child3_dob','child4_dob','child5_dob'],axis=1)

dfpolicy=dfpolicy.fillna(0)

dfpolicy.loc[(dfpolicy['child1_gender']=='M')|(dfpolicy['child1_gender']=='F') ,'child1_gender']=1
dfpolicy.loc[(dfpolicy['child2_gender']=='M')|(dfpolicy['child2_gender']=='F') ,'child2_gender']=1
dfpolicy.loc[(dfpolicy['child3_gender']=='M')|(dfpolicy['child3_gender']=='F') ,'child3_gender']=1
dfpolicy.loc[(dfpolicy['child4_gender']=='M')|(dfpolicy['child4_gender']=='F') ,'child4_gender']=1
dfpolicy.loc[(dfpolicy['child5_gender']=='M')|(dfpolicy['child5_gender']=='F') ,'child5_gender']=1

no_of_child = ['child1_gender','child2_gender','child3_gender','child4_gender','child5_gender']
dfpolicy["No_of_child"] = dfpolicy[no_of_child].sum(axis=1)

dfpolicy = dfpolicy.drop(['child1_gender','child2_gender','child3_gender','child4_gender','child5_gender'],axis=1)

dfpolicy.loc[(dfpolicy['spouse_gender']=='M')|(dfpolicy['spouse_gender']=='F') ,'spouse_gender']=1

dfpolicy.loc[dfpolicy['spouse_smoker_flag']=='Y','spouse_smoker_flag']=1
dfpolicy.loc[dfpolicy['spouse_smoker_flag']=='N','spouse_smoker_flag']=0



rider = ['rider1_prem','rider2_prem','rider3_prem','rider4_prem','rider5_prem','rider6_prem','rider7_prem','rider8_prem','rider9_prem','rider10_prem']
dfpolicy["Target"] = dfpolicy[rider].sum(axis=1)

dfpolicy.loc[dfpolicy['Target']>0,'Riders']=1
dfpolicy.loc[dfpolicy['Target']==0,'Riders']=0

dfpolicy.loc[dfpolicy['rider1_sum_assuared']>0,'Rider_1']=1
dfpolicy.loc[dfpolicy['rider1_sum_assuared']==0,'Rider_1']=0

dfpolicy.loc[dfpolicy['rider2_sum_assuared']>0,'Rider_2']=1
dfpolicy.loc[dfpolicy['rider2_sum_assuared']==0,'Rider_2']=0

dfpolicy.loc[dfpolicy['rider3_sum_assuared']>0,'Rider_3']=1
dfpolicy.loc[dfpolicy['rider3_sum_assuared']==0,'Rider_3']=0

dfpolicy.loc[dfpolicy['rider4_sum_assuared']>0,'Rider_4']=1
dfpolicy.loc[dfpolicy['rider4_sum_assuared']==0,'Rider_4']=0

dfpolicy.loc[dfpolicy['rider5_sum_assuared']>0,'Rider_5']=1
dfpolicy.loc[dfpolicy['rider5_sum_assuared']==0,'Rider_5']=0

dfpolicy.loc[dfpolicy['rider6_sum_assuared']>0,'Rider_6']=1
dfpolicy.loc[dfpolicy['rider6_sum_assuared']==0,'Rider_6']=0

dfpolicy.loc[dfpolicy['rider7_sum_assuared']>0,'Rider_7']=1
dfpolicy.loc[dfpolicy['rider7_sum_assuared']==0,'Rider_7']=0

dfpolicy.loc[dfpolicy['rider8_sum_assuared']>0,'Rider_8']=1
dfpolicy.loc[dfpolicy['rider8_sum_assuared']==0,'Rider_8']=0

dfpolicy.loc[dfpolicy['rider9_sum_assuared']>0,'Rider_9']=1
dfpolicy.loc[dfpolicy['rider9_sum_assuared']==0,'Rider_9']=0

dfpolicy.loc[dfpolicy['rider10_sum_assuared']>0,'Rider_10']=1
dfpolicy.loc[dfpolicy['rider10_sum_assuared']==0,'Rider_10']=0

no_of_policies = ['Rider_1','Rider_2','Rider_3','Rider_4','Rider_5','Rider_6','Rider_7','Rider_8','Rider_9','Rider_10']
dfpolicy["No_of_policies"] = dfpolicy[no_of_policies].sum(axis=1)
dfpolicy = dfpolicy.drop(['Rider_1','Rider_2','Rider_3','Rider_4','Rider_5','Rider_6','Rider_7','Rider_8','Rider_9','Rider_10'],axis=1)

frequencies = dfpolicy['agent_code'].value_counts()
dfpolicy['agent_count'] = dfpolicy['agent_code'].apply(lambda x: frequencies[x])

dfpolicy = dfpolicy.drop('agent_code',axis=1)

from sklearn.cluster import KMeans


#wcss=[]
#for i in range(1, 11):
   # kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
  #  kmeans.fit(x)
 #   wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
#plt.plot(range(1, 11), wcss)
#plt.title('The elbow method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS') #within cluster sum of squares
#plt.show()



#dfpolicy.to_csv("New_policy_data2.csv")

dfpolicy['main_holder_gender']=pd.get_dummies(dfpolicy['main_holder_gender'],drop_first=True)
dfpolicy['main_holder_smoker_flag']=pd.get_dummies(dfpolicy['main_holder_smoker_flag'],drop_first=True)

dfpolicy.loc[dfpolicy['policy_payment_mode']=='H','policy_payment_mode']=0
dfpolicy.loc[dfpolicy['policy_payment_mode']=='M','policy_payment_mode']=1
dfpolicy.loc[dfpolicy['policy_payment_mode']=='Q','policy_payment_mode']=2
dfpolicy.loc[dfpolicy['policy_payment_mode']=='S','policy_payment_mode']=3
dfpolicy.loc[dfpolicy['policy_payment_mode']=='Y','policy_payment_mode']=4

dfpolicy.loc[dfpolicy['product_code']=='ED001','product_code']=0
dfpolicy.loc[dfpolicy['product_code']=='HE001','product_code']=1
dfpolicy.loc[dfpolicy['product_code']=='IN001','product_code']=2
dfpolicy.loc[dfpolicy['product_code']=='PR001','product_code']=3
dfpolicy.loc[dfpolicy['product_code']=='RE001','product_code']=4


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x=dfpolicy.iloc[:, [32,36,37]]

kmeans= KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(x)

dfpolicy['target']=y_kmeans
X=dfpolicy.drop('target',axis=1)

#import seaborn as sns
#corrmat = dfpolicy.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#g=sns.heatmap(dfpolicy[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#from sklearn.ensemble import ExtraTreesClassifier
#import matplotlib.pyplot as plt
#model = ExtraTreesClassifier()
#model.fit(X,y_kmeans)

#print(model.feature_importances_) 

#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()

dfpolicy['policy_payment_mode']=dfpolicy['policy_payment_mode'].astype(int)
dfpolicy['main_holder_gender']=dfpolicy['main_holder_gender'].astype(int)
dfpolicy['main_holder_smoker_flag']=dfpolicy['main_holder_smoker_flag'].astype(int)
dfpolicy['spouse_smoker_flag']=dfpolicy['spouse_smoker_flag'].astype(int)
dfpolicy['product_code']=dfpolicy['product_code'].astype(int)

dfpolicy2= dfpolicy.drop(['agent_count','No_of_policies','Target','Duration','policy_snapshot_as_on','total_sum_assuared','premium_value'],axis=1)

train_img, test_img, train_lbl, test_lbl = train_test_split(dfpolicy2.drop(labels=['target'],axis=1), dfpolicy2.target, test_size=1/7.0, random_state=0)

test_img = test_img.to_numpy()
train_lbl = train_lbl.to_numpy()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(train_img, train_lbl)

pre=model.predict(test_img)
tt=model.predict_proba(test_img)

