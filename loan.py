#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[3]:


df=pd.read_csv('C:/Users/SIDDHANT DESHMUKH/Desktop/ml/loan.csv',low_memory=False)


# In[4]:


df = df[df.loan_status != 'Current']
df = df[df.loan_status != 'In Grace Period']
df = df[df.loan_status != 'Late (16-30 days)']
df = df[df.loan_status != 'Late (31-120 days)']
df = df[df.loan_status != 'Does not meet the credit policy. Status:Fully Paid']
df = df[df.loan_status != 'Does not meet the credit policy. Status:Charged Off']
df = df[df.loan_status != 'Issued']


# In[5]:


df['loan_status'] = df['loan_status'].replace({'Charged Off':'Default'})
df['loan_status'].value_counts()
df.head()


# In[6]:


df_null = pd.DataFrame({'Count': df.isnull().sum(), 'Percent': 100*df.isnull().sum()/len(df)})


# In[7]:


df_null[df_null['Count']>0]


# In[8]:


df1 = df.dropna(axis=1, thresh=int(0.80*len(df)))
df1.columns


# In[9]:


df1=df1.drop(['id','member_id','sub_grade','emp_title','url','verification_status','title','zip_code','earliest_cr_line','inq_last_6mths','open_acc'],axis=1)


# In[10]:


df1=df1.drop(['pub_rec','revol_bal','revol_util','initial_list_status','out_prncp','out_prncp_inv','out_prncp_inv','total_pymnt_inv','total_rec_prncp','total_rec_int'],axis=1)


# In[11]:


df1.columns


# In[12]:


df[['total_rec_int','total_rec_prncp','total_pymnt']].head()


# In[13]:


df[['last_pymnt_amnt','installment']].head()


# In[14]:


df1=df1.drop(['total_rec_late_fee','collection_recovery_fee','grade'],axis=1)


# In[15]:


df1.columns


# In[16]:


np.unique(df.values[:,-4],return_counts=True)


# In[17]:


df[['total_rec_late_fee','collection_recovery_fee']].head()


# In[18]:


df1.columns


# In[19]:


df1=df1.drop(['collections_12_mths_ex_med','policy_code','loan_amnt'],axis=1)


# In[20]:


df1=df1.drop(['acc_now_delinq'],axis=1)


# In[21]:


df1=df1.drop(['last_credit_pull_d','last_pymnt_amnt'],axis=1)


# In[22]:


df1=df1.drop(['last_pymnt_d'],axis=1)


# In[23]:


df1=df1.drop(['funded_amnt_inv'],axis=1)


# In[24]:


df1.shape


# In[25]:


df1=df1.dropna(subset=['annual_inc'])


# In[26]:


df_null = pd.DataFrame({'Count': df.isnull().sum(), 'Percent': 100*df.isnull().sum()/len(df)})
df_null


# In[27]:


df1.shape


# In[28]:


df1['loan_status'].unique()


# In[29]:


df1.reset_index()


# In[30]:


df2=df1.drop(['emp_length'],axis=1)


# In[31]:


df2.columns
np.unique(df2['home_ownership'],return_counts=True)
df2 = df2[df2['home_ownership']!='ANY']


# In[32]:


np.unique(df2['home_ownership'],return_counts=True)


# In[33]:


#df2.loc[(df2['home_ownership'] == 'OTHER') & (df['loan_status'] == 'Default')]
df2 = df2[df2['home_ownership']!='OTHER']
df2 = df2[df2['home_ownership']!='NONE']


# In[34]:


df2.head()


# In[35]:


df2 = df2.drop('addr_state', axis = 1)


# In[36]:


df2 = df2.drop('issue_d', axis = 1)


# In[37]:


np.unique(df2['application_type'],return_counts=True)


# In[38]:


df2 = df2.drop('application_type', axis = 1)


# In[39]:


df2.columns


# In[40]:


df2.head()


# In[41]:


dummy_ho = pd.get_dummies(df2['home_ownership'])


# In[42]:


dummy_ho.head()


# In[43]:


df2 = pd.concat([df2,dummy_ho],axis = 1)


# In[44]:


dummy_purpose = pd.get_dummies(df2['purpose'])


# In[45]:


dummy_purpose.head()


# In[46]:


df2 = pd.concat([df2,dummy_purpose],axis = 1)


# In[47]:


df2 = df2.drop(['home_ownership','purpose'],axis = 1)


# In[49]:


df2['pymnt_plan'] = df2['pymnt_plan'].replace({'y':'1'})


# In[50]:


df2['pymnt_plan'] = df2['pymnt_plan'].replace({'n':'0'})


# In[52]:


df2['loan_status'] = df2['loan_status'].replace({'Fully Paid':'1'})
df2['loan_status'] = df2['loan_status'].replace({'Default':'0'})


# In[53]:


df2.shape


# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix


# In[56]:


df2.columns


# In[58]:


X=df2[['funded_amnt', 'term', 'int_rate', 'installment', 'annual_inc',
       'pymnt_plan', 'dti', 'delinq_2yrs', 'total_acc',
       'total_pymnt', 'recoveries', 'MORTGAGE', 'OWN', 'RENT', 'car',
       'credit_card', 'debt_consolidation', 'educational', 'home_improvement',
       'house', 'major_purchase', 'medical', 'moving', 'other',
       'renewable_energy', 'small_business', 'vacation', 'wedding']]
y=df2['loan_status']


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf=DecisionTreeClassifier(criterion="entropy")


# In[60]:


reg=clf.fit(X_train,y_train)
prediction=reg.predict(X_test)


# In[61]:


accuracy_score(y_test,prediction)#now this is some bullshit!!!you know why because the test cases are imballenced......Oh Crap!!


# In[64]:


result=confusion_matrix(y_test,prediction)
print(result)


# In[65]:


tprecision=(result[0][0]/(result[0][1]+result[0][0]))*100


# In[66]:


trecall=result[0][0]*100/(result[0][0]+result[1][0])


# In[67]:


fscorep=2*(tprecision*trecall)/(tprecision+trecall)
fscorep


# In[68]:


fprecision=result[1][1]/(result[1][0]+result[1][1])


# In[69]:


frecall=result[1][1]/(result[0][1]+result[1][1])


# In[70]:


fscoren=2*(fprecision*frecall)/(fprecision+frecall)
fscoren*100


# In[71]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
X_train_new,y_train_new=sm.fit_sample(X_train,y_train)
clf=DecisionTreeClassifier(criterion="entropy")
reg=clf.fit(X_train_new,y_train_new)
prediction=reg.predict(X_test)
accuracy_score(y_test,prediction)


# In[72]:


np.unique(y_train_new,return_counts=True)


# In[73]:


result=confusion_matrix(y_test,prediction)
result


# In[74]:


13533/(13533+352)


# In[75]:


(13533+61999)/(13533+352+61999+352)


# In[ ]:




