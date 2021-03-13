import pandas as pd # import library
import numpy as np # import lib
import statistics as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics

df=pd.read_csv(r"E:\Academic\Data storm//Hotel-A-train.csv" )
dfval = pd.read_csv(r"E:\Academic\Data storm//Hotel-A-validation.csv")
# train set
df['Expected_checkin'] = pd.to_datetime(df['Expected_checkin'],infer_datetime_format=True)
df['Expected_checkout'] = pd.to_datetime(df['Expected_checkout'],infer_datetime_format=True)
df['Booking_date'] = pd.to_datetime(df['Booking_date'],infer_datetime_format=True)

df['Days'] = df['Expected_checkout']-df['Booking_date']
#df['Months'] = df.Expected_checkin.dt.month
df['Days'] = df.Days.dt.days
df=df[df['Days']<350]
#dfdf2=df.
#df5['New']=df['Reservation-id']
df = df.drop(['Reservation-id','Expected_checkin','Expected_checkout','Booking_date'],axis=1)

df.loc[df['Adults']==1,'Adults']=0
df.loc[df['Adults']==2,'Adults']=1
df.loc[df['Adults']==3,'Adults']=2
df.loc[(df['Adults']==4)|(df['Adults']==5) ,'Adults']=3

df.loc[df['Babies']==0,'Babies']=0
df.loc[(df['Babies']==1)|(df['Babies']==2) ,'Babies']=1

df.loc[(df['Age']>=18) &(df['Age']<=30),'Age']=0
df.loc[(df['Age']>=31) &(df['Age']<=44),'Age']=1
df.loc[(df['Age']>=45) &(df['Age']<=57),'Age']=2
df.loc[df['Age']>=58,'Age']=3

df.loc[df['Children']==1,'Children']='c1'
df.loc[df['Children']==2,'Children']='c2'
df.loc[df['Children']==3,'Children']='c3'

#df.loc[(df['Months']>=1) &(df['Months']<=3),'Months']=1
#df.loc[(df['Months']>=4) &(df['Months']<=7),'Months']=2
#df.loc[df['Months']>=8,'Months']=3


df.loc[(df['Room_Rate']>=100) &(df['Room_Rate']<=140),'Room_Rate']=0
df.loc[(df['Room_Rate']>=141) &(df['Room_Rate']<=180),'Room_Rate']=1
df.loc[(df['Room_Rate']>=181) &(df['Room_Rate']<=220),'Room_Rate']=2
df.loc[df['Room_Rate']>=221,'Room_Rate']=3

df.loc[df['Discount_Rate']==0,'Discount_Rate']=0
df.loc[(df['Discount_Rate']>=5) &(df['Discount_Rate']<=15),'Discount_Rate']=1
df.loc[(df['Discount_Rate']>=20) &(df['Discount_Rate']<=25),'Discount_Rate']=2
df.loc[df['Discount_Rate']>=30,'Discount_Rate']=3

df.loc[df['Deposit_type']=='No Deposit','Deposit_type']=0
df.loc[df['Deposit_type']=='Refundable','Deposit_type']=1
df.loc[df['Deposit_type']=='Non-Refundable','Deposit_type']=2

df.loc[df['Income']=='<25K','Income']=0
df.loc[df['Income']=='25K --50K','Income']=1
df.loc[df['Income']=='50K -- 100K','Income']=2
df.loc[df['Income']=='>100K','Income']=3

money = ['Deposit_type', 'Income']
df["Money"] = df[money].sum(axis=1)

dfnew=df.drop(['Gender','Ethnicity','Adults','Income','Country_region','Visted_Previously','Meal_Type','Deposit_type','Booking_channel','Discount_Rate','Hotel_Type','Reservation_Status','Room_Rate'],axis=1)


# Validation set
dfval['Expected_checkin'] = pd.to_datetime(dfval['Expected_checkin'],infer_datetime_format=True)
dfval['Expected_checkout'] = pd.to_datetime(dfval['Expected_checkout'],infer_datetime_format=True)
dfval['Booking_date'] = pd.to_datetime(dfval['Booking_date'],infer_datetime_format=True)


dfval['Days'] = dfval['Expected_checkout']-dfval['Booking_date']
#dfval['Months'] = dfval.Expected_checkin.dt.month
dfval['Days'] = dfval.Days.dt.days
dfval=dfval[dfval['Days']<350]

dfval = dfval.drop(['Reservation-id','Gender','Expected_checkin','Expected_checkout','Booking_date'],axis=1)


dfval.loc[dfval['Adults']==1,'Adults']=0
dfval.loc[dfval['Adults']==2,'Adults']=1
dfval.loc[dfval['Adults']==3,'Adults']=2
dfval.loc[(dfval['Adults']==4)|(dfval['Adults']==5) ,'Adults']=3

dfval.loc[dfval['Babies']==0,'Babies']=0
dfval.loc[(dfval['Babies']==1) | (dfval['Babies']==2) ,'Babies']=1

dfval.loc[(dfval['Age']>=18) &(dfval['Age']<=30),'Age']=0
dfval.loc[(dfval['Age']>=31) &(dfval['Age']<=44),'Age']=1
dfval.loc[(dfval['Age']>=45) &(dfval['Age']<=57),'Age']=2
dfval.loc[dfval['Age']>=58,'Age']=4

dfval.loc[dfval['Deposit_type']=='No Deposit','Deposit_type']=0
dfval.loc[dfval['Deposit_type']=='Refundable','Deposit_type']=1
dfval.loc[dfval['Deposit_type']=='Non-Refundable','Deposit_type']=2

dfval.loc[(dfval['Room_Rate']>=100) &(dfval['Room_Rate']<=140),'Room_Rate']=0
dfval.loc[(dfval['Room_Rate']>=141) &(dfval['Room_Rate']<=180),'Room_Rate']=1
dfval.loc[(dfval['Room_Rate']>=181) &(dfval['Room_Rate']<=220),'Room_Rate']=2
dfval.loc[dfval['Room_Rate']>=221,'Room_Rate']=3

dfval.loc[dfval['Discount_Rate']==0,'Discount_Rate']=0
dfval.loc[(dfval['Discount_Rate']>=5) &(dfval['Discount_Rate']<=15),'Discount_Rate']=1
dfval.loc[(dfval['Discount_Rate']>=20) &(dfval['Discount_Rate']<=25),'Discount_Rate']=2
dfval.loc[dfval['Discount_Rate']>=30,'Discount_Rate']=3

dfval.loc[dfval['Children']==1,'Children']='c1'
dfval.loc[dfval['Children']==2,'Children']='c2'
dfval.loc[dfval['Children']==3,'Children']='c3'

dfval.loc[dfval['Income']=='<25K','Income']=0
dfval.loc[dfval['Income']=='25K --50K','Income']=1
dfval.loc[dfval['Income']=='50K -- 100K','Income']=2
dfval.loc[dfval['Income']=='>100K','Income']=3

money = ['Deposit_type','Income']
dfval["Money"] = dfval[money].sum(axis=1)

dfnewval=dfval.drop(['Ethnicity','Income','Adults','Country_region','Hotel_Type','Meal_Type','Visted_Previously','Deposit_type','Booking_channel','Discount_Rate','Room_Rate','Reservation_Status'],axis=1)

# Train
X_train = dfnew.copy()
y_train = df.pop('Reservation_Status')
X_valid = dfnewval.copy()
y_valid = dfval.pop('Reservation_Status')

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

from sklearn.preprocessing import OneHotEncoder
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

from sklearn.preprocessing import StandardScaler 
# create the scaler 
ss = StandardScaler() 
# apply the scaler to the dataframe subset 
subset_scaled = ss.fit_transform(OH_X_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(leaf_size=2,n_neighbors =1,p=1) 
model=knn.fit(OH_X_train, y_train)
y_pred=model.predict(OH_X_valid)
print(knn.score(OH_X_valid, y_valid))

from sklearn.metrics import f1_score
print(f1_score(y_valid, y_pred, average='macro'))



