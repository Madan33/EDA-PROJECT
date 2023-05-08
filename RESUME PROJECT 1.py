#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# ## reading and understanding the Data

# In[3]:


Madhan=pd.read_csv(r"C:\Users\smkon\Downloads\application_data.csv\application_data.csv")


# In[4]:


Madhan


# In[5]:


Madhan1=pd.read_csv(r"C:\Users\smkon\Downloads\previous_application.csv\previous_application.csv")


# In[6]:


Madhan1


# ## by default its show top five columns

# In[7]:


Madhan.head()


# In[8]:


Madhan[["SK_ID_CURR"]].head()


# ## by default its show Bottom five columns

# In[9]:


Madhan.tail()


# ## Database column type

# In[10]:


Madhan.columns,Madhan1.columns


# ## inspect DataFrame

# In[11]:


print("database dimension-SMK:",Madhan.shape)
print("database dimension-SMK:",Madhan.size)
print("database dimension-SMK:",Madhan1.shape)
print("database dimension-SMK:",Madhan1.size)


# In[12]:


Madhan1.shape


# ## it shows total information about Dataset

# In[13]:


Madhan.info(verbose=True)


# ## checking the numeric variables of the dataframe

# In[14]:


Madhan.describe()


# In[15]:


Madhan.describe().transpose()


# In[16]:


kumar=Madhan[Madhan1["SK_ID_CURR"]>=50]


# In[17]:


kumar


# In[18]:


len(kumar)


# In[19]:


Madhan.iloc[0::4].std()


# In[20]:


Madhan.corr()


# In[21]:


Madhan.min()


# In[22]:


Madhan.mode()


# In[23]:


Madhan.min()


# In[24]:


Madhan.std()


# In[25]:


Madhan.median


# In[26]:


Madhan.max()


# ## droping the unncessary columns from given dataset

# In[27]:


Madhan.drop(labels=Madhan,axis=1,inplace=True)


# In[28]:


Madhan


# In[29]:


SMK=Madhan1["SK_ID_CURR"].tolist()
SMK


# ## data cleaning nd misssing value

# In[30]:


import missingno as mn
mn.matrix(Madhan1)


# ## % Null value  in Each column

# In[31]:


round(Madhan1.isnull().sum() / Madhan1.shape[0]*100.00,2)


# In[32]:


Madhan1.isna().any()


# In[33]:



null_applicationDF = pd.DataFrame((Madhan1.isnull().sum())*100/Madhan1.shape[0]).reset_index()
null_applicationDF.columns = ['Column Name', 'Null Values Percentage']
fig = plt.figure(figsize=(18,6))
ax = sns.pointplot(x="Column Name",y="Null Values Percentage",data=null_applicationDF,color='blue')
plt.xticks(rotation =90,fontsize =7)
ax.axhline(40, ls='--',color='red')
plt.title("Percentage of Missing values in application data")
plt.ylabel("Null Values PERCENTAGE")
plt.xlabel("COLUMNS")
plt.show()


# ## more thaat or equal to 40% empty rows columns

# In[34]:


Kumar=null_applicationDF[null_applicationDF["Null Values Percentage"]>40]


# In[35]:


Kumar


# In[36]:


len(Kumar)


# ## Analyze the unnecessary columns in Madhan1

# In[37]:



Source = Madhan1[["AMT_ANNUITY","AMT_APPLICATION","AMT_GOODS_PRICE","AMT_CREDIT"]]
fig = plt.figure(figsize=(18,6))
source_corr = Source.corr()
ax = sns.heatmap(source_corr,
            xticklabels=source_corr.columns,
            yticklabels=source_corr.columns,
            annot = True,
            cmap ="RdYlGn")


#  ### create a list of columns that needs to be dropped including the columns with >40% null values

# In[38]:


Kumar1= Kumar["Column Name"].tolist()+ ["RATE_DOWN_PAYMENT","NAME_TYPE_SUITE"] 
# as EXT_SOURCE_1 column is already included in nullcol_40_application 
len(Kumar1)


# In[ ]:





# In[39]:



Source = Madhan1[["AMT_ANNUITY","AMT_APPLICATION","AMT_GOODS_PRICE","AMT_CREDIT","HOUR_APPR_PROCESS_START","CNT_PAYMENT"]]
fig = plt.figure(figsize=(18,6))
source_corr = Source.corr()
ax = sns.heatmap(source_corr,
            xticklabels=source_corr.columns,
            yticklabels=source_corr.columns,
            annot = True,
            cmap ="RdYlGn")


# In[40]:


Madhan1.drop(labels=Kumar1,axis=1)


# ## getting the columns which has mopre thaat 40% unknmown

# In[41]:


Maddy=Kumar["Column Name"].tolist()
Maddy


# In[42]:


len(Maddy)


#  ## Listing down columns which are not needed

# In[43]:



Maddy1 = ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']


# In[44]:


Maddy1=Maddy+Maddy1


# In[45]:


Maddy1


# In[46]:


len(Maddy1),len(Maddy)


# In[47]:


#Madhan1.Remove('AMT_APPLICATION') 
#Unwanted_application = Unwanted_application + contact_col
len(Madhan1)


# ### Getting the 11 columns which has more than 40% unknown

# In[48]:


Unwanted_previous = Kumar["Column Name"].tolist()
Unwanted_previous


# ## converting negaive days to postive days

# In[49]:



SMK= ["AMT_APPLICATION","AMT_CREDIT","AMT_GOODS_PRICE"]

for col in SMK:
    Madhan1[col] = abs (Madhan1[col])


# In[50]:


SMK


# ## binning numerical column to creat a categorical column

# In[51]:


Madhan1["AMT_APPLICATION"]=Madhan1["AMT_APPLICATION"]/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,11]
slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']

Madhan1["DAYS_FIRST_DUE"]=pd.cut(Madhan1["DAYS_FIRST_DUE"],bins,labels=slot)


# In[52]:


Madhan1["DAYS_FIRST_DUE"].value_counts(normalize=True)*1000


# In[53]:


Madhan1["AMT_CREDIT"]=Madhan1["AMT_CREDIT"]/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,11]
slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']

Madhan1["CNT_PAYMENT"]=pd.cut(Madhan1["CNT_PAYMENT"],bins,labels=slot)


# In[54]:


Madhan1["CNT_PAYMENT"].value_counts(normalize=True)*100


# In[55]:


Madhan1["AMT_GOODS_PRICE"]=Madhan1["AMT_GOODS_PRICE"]/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,11]
slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']

Madhan1["DAYS_LAST_DUE"]=pd.cut(Madhan1["DAYS_LAST_DUE"],bins,labels=slot)


# In[56]:


Madhan1["DAYS_LAST_DUE"].value_counts(normalize=True)*100


# ## checking the number of unique values each column posses to 

# In[57]:


Madhan1.nunique().sort_values()


# ## Inspecting the column type if type are in correct data type using the above resuly:

# In[58]:


Madhan1.info()


# In[59]:


Madhan=pd.read_csv(r"C:\Users\smkon\Downloads\application_data.csv\application_data.csv")


# In[60]:


Madhan


# ## converting of object and numerical columns to categorical columns

# In[61]:


categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY' ]
                      
for col in categorical_columns:
    Madhan[col] =pd.Categorical(Madhan[col])


# ## inspecting the column types if the above convesion is reflected

# In[62]:


Madhan.info()


# ## checking the number of umique value each column pposses tp identify categorical columns 

# In[63]:


Madhan.nunique().sort_values()


# ## converting negative day to postive day

# In[64]:


Madhan1["AMT_APPLICATION"]=abs(Madhan1["AMT_APPLICATION"])


# In[65]:


Madhan1["AMT_APPLICATION"].value_counts(normalize=True)*100


# ### Converting Categorical columns from Object to categorical 

# In[66]:


Catgorical_col_p = ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                    'NAME_CONTRACT_TYPE']

for col in Catgorical_col_p:
    Madhan1[col] =pd.Categorical(Madhan1[col])


# In[67]:


Madhan1.columns


# In[68]:


Madhan1.info()


#  ## checking the null value % of each column in applicationDF dataframe

# In[69]:



round(Madhan.isnull().sum() / Madhan.shape[0] * 100.00,2)


# ## impute caterogical vriable "AMT_ANNUITY" which has lower null percentage(0.42%)with the most frequent category using mode()[0]

# In[70]:


Madhan["AMT_ANNUITY"].describe()


# In[71]:


Madhan["AMT_ANNUITY"].fillna(Madhan["AMT_ANNUITY"].mode()[0],inplace=True)


# ### Impute categorical variable 'OCCUPATION_TYPE' which has higher null percentage(31.35%) with a new category as assigning to any existing category might influence the analysis:

# In[72]:


## Madhan1["AMT_ANNUITY"] = Madhan1["AMT_ANNUITY"].cat.add_categoricaal("unknown")
## Madhan1["AMT_ANNUITY"].fillna("unknown",inplace="True")


# In[73]:


Madhan[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# In[74]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']
for col in amount:
    Madhan[col].fillna(Madhan[col].median(),inplace = True)


# In[75]:


Catgorical_col_p = ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                    'NAME_CONTRACT_TYPE']

for col in Catgorical_col_p:
    Madhan1[col] =pd.Categorical(Madhan1[col])


# # checking the null value % of each column in previousDF dataframe

# In[76]:



round(Madhan1.isnull().sum() / Madhan1.shape[0] * 100.00,2)


# ### Impute AMT_ANNUITY with median as the distribution is greatly skewed:

# In[77]:


plt.figure(figsize=(6,6))
sns.kdeplot(Madhan1['AMT_ANNUITY'])
plt.show()


# In[78]:


Madhan1['AMT_ANNUITY'].fillna(Madhan1['AMT_ANNUITY'].median(),inplace = True)


# ### Impute AMT_GOODS_PRICE with mode as the distribution is closely similar

# In[79]:


plt.figure(figsize=(6,6))
sns.kdeplot(Madhan1['AMT_GOODS_PRICE'][pd.notnull(Madhan1['AMT_GOODS_PRICE'])])
plt.show()


# #### There are several peaks along the distribution. Let's impute using the mode, mean and median and see if the distribution is still about the same.

# In[80]:


SMK = pd.DataFrame() # new dataframe with columns imputed with mode, median and mean
SMK['AMT_GOODS_PRICE_mode'] = Madhan1['AMT_GOODS_PRICE'].fillna(Madhan1['AMT_GOODS_PRICE'].mode()[0])
SMK['AMT_GOODS_PRICE_median'] = Madhan1['AMT_GOODS_PRICE'].fillna(Madhan1['AMT_GOODS_PRICE'].median())
SMK['AMT_GOODS_PRICE_mean'] = Madhan1['AMT_GOODS_PRICE'].fillna(Madhan1['AMT_GOODS_PRICE'].mean())

cols = ['AMT_GOODS_PRICE_mode', 'AMT_GOODS_PRICE_median','AMT_GOODS_PRICE_mean']

plt.figure(figsize=(18,10))
plt.suptitle('Distribution of Original data vs imputed data')
plt.subplot(221)
sns.distplot(Madhan1['AMT_GOODS_PRICE'][pd.notnull(Madhan1['AMT_GOODS_PRICE'])]);
for i in enumerate(cols): 
    plt.subplot(2,2,i[0]+2)
    sns.distplot(SMK[i[1]])


# ### The original distribution is closer with the distribution of data imputed with mode in this case

# In[81]:


Madhan1['AMT_GOODS_PRICE'].fillna(Madhan1['AMT_GOODS_PRICE'].mode()[0], inplace=True)


# #### Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS for these indicate that most of these loans were not started

# In[82]:


Madhan1.loc[Madhan1['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# In[83]:


Madhan1['AMT_GOODS_PRICE'].fillna(0,inplace = True)


# ### Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS for these indicate that most of these loans were not started

# In[84]:


Madhan1.loc[Madhan1['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# # checking the null value % of each column in previousDF dataframe

# In[85]:



round(Madhan1.isnull().sum() / Madhan1.shape[0] * 100.00,2)


#  ### Finding outlier information in applicationDF

# In[86]:


plt.figure(figsize=(22,10))
app_outlier_col_1 = ["AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_DOWN_PAYMENT","AMT_GOODS_PRICE"]
app_outlier_col_2 = ["AMT_CREDIT","AMT_DOWN_PAYMENT"]
for i in enumerate(app_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=Madhan1[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(app_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=Madhan1[i[1]])
    plt.title(i[1])
    plt.ylabel("")


# ### We can see the stats for these columns below as well.

# In[87]:


Madhan1[["AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_DOWN_PAYMENT","AMT_GOODS_PRICE"]].describe()


# In[88]:


Madhan=pd.read_csv(r"C:\Users\smkon\Downloads\application_data.csv\application_data.csv")


# In[89]:


Madhan


# ## Finding outlier information in Madhan

# In[90]:


plt.figure(figsize=(22,10))
prev_outlier_col_1 = ["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT"]

prev_outlier_col_2 = ["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK"]
for i in enumerate(prev_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=Madhan[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(prev_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=Madhan[i[1]])
    plt.title(i[1])
    plt.ylabel("") 


# In[91]:


Madhan1[['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'SELLERPLACE_AREA','CNT_PAYMENT','DAYS_DECISION']].describe()


# ## Finding outlier information in Madhan1

# In[101]:


plt.figure(figsize=(22,8))

prev_outlier_col_1 = ['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE']
prev_outlier_col_2 = ['SK_ID_CURR','AMT_GOODS_PRICE']
for i in enumerate(prev_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=Madhan[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(prev_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=Madhan[i[1]])
    plt.title(i[1])
    plt.ylabel("") 


# ### We can see the stats for these columns below as well.

# In[94]:


Madhan1[['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'SELLERPLACE_AREA','CNT_PAYMENT','DAYS_DECISION']].describe()


# ### Imbalance Analysis

# In[95]:


alance = Madhan1["AMT_APPLICATION"].value_counts().reset_index()

plt.figure(figsize=(10,4))
x= ["AMT_APPLICATION","AMT_CREDIT"]
sns.barplot(x="AMT_APPLICATION",data =Madhan1,palette= ['g','r'])
plt.xlabel("Loan Repayment Status")
plt.ylabel("Count of Repayers & Defaulters")
plt.title("Imbalance Plotting")
plt.show()


# In[102]:


count_0 = alance .iloc[0]["AMT_APPLICATION"]
count_1 = alance .iloc[1]["AMT_APPLICATION"]
count_0_perc = round(count_0/(count_0+count_1)*100,2)
count_1_perc = round(count_1/(count_0+count_1)*100,2)

print('Ratios of imbalance in percentage with respect to Repayer and Defaulter datas are: %.2f and %.2f'%(count_0_perc,count_1_perc))
print('Ratios of imbalance in relative with respect to Repayer and Defaulter datas is %.2f : 1 (approx)'%(count_0/count_1))


# ##  function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# # This function will create two subplots: 
# # 1. Count plot of categorical column w.r.t TARGET; 
# # 2. Percentage of defaulters within column

# In[103]:


def SMK1(feature,ylog=False,label_rotation=False,horizontal_layout=True):
    temp = Madhan[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = Madhan[[feature,"TARGET"]].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=Madhan,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
    
    # Define common styling
    ax1.set_title(feature, fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'}) 
    ax1.legend(['Repayer','Defaulter'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of Defaulters [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.show();


# # function for plotting repetitive countplots in bivariate categorical analysis

# In[104]:



def bivariate_bar(x,y,df,hue,figsize):
    
    plt.figure(figsize=figsize)
    sns.barplot(x=x,
                  y=y,
                  data=Madhan1, 
                  hue=hue, 
                  palette =['g','r'])     
        
    # Defining aesthetics of Labels and Title of the plot using style dictionaries
    plt.xlabel(x,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.ylabel(y,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.title(col, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.xticks(rotation=90, ha='right')
    plt.legend(labels = ['Repayer','Defaulter'])
    plt.show()


# ### function for plotting repetitive countplots in univariate categorical analysis on the merged df

# ###  function for plotting repetitive countplots in bivariate categorical analysis

# In[105]:


def bivariate_rel(x,y,data, hue, kind, palette, legend,figsize):
    
    plt.figure(figsize=figsize)
    sns.relplot(x=x, 
                y=y, 
                data=Madhan, 
                hue="TARGET",
                kind=kind,
                palette = ['g','r'],
                legend = False)
    plt.legend(['Repayer','Defaulter'])
    plt.xticks(rotation=90, ha='right')
    plt.show()


# ###  function for plotting repetitive rel plots in bivaritae numerical analysis on applicationDF

# In[106]:


def univariate_merged(col,df,hue,palette,ylog,figsize):
    plt.figure(figsize=figsize)
    ax=sns.countplot(x=col, 
                  data=Madhan1,
                  hue= hue,
                  palette= palette,
                  order=df[col].value_counts().index)
    

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})     
    else:
        plt.ylabel("Count",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})       

    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


# ###  Function to plot point plots on merged dataframe

# In[107]:



def merged_pointplot(x,y):
    plt.figure(figsize=(8,4))
    sns.pointplot(x=x, 
                  y=y, 
                  hue="TARGET", 
                  data=loan_process_df,
                  palette =['g','r'])


# In[108]:


SMK1


# In[109]:


Madhan


# ###  Checking the contract type based on loan repayment status

# In[110]:


SMK1("NAME_CONTRACT_TYPE",True)


# # Inferences:
# Contract type: Revolving loans are just a small fraction (10%) from the total number of loans; in the same time, a larger amount of Revolving loans, comparing with their frequency, are not repaid.

# ###  Checking the type of Gender on loan repayment status

# In[111]:



SMK1('CODE_GENDER')


# # Inferences:
# * The number of female clients is almost double the number of male clients. Based on the percentage of defaulted credits, males have a higher chance of not returning their loans (~10%), comparing with women (~7%)

# ###  Analyzing Family status based on loan repayment status

# In[112]:


SMK1("NAME_FAMILY_STATUS",False,True,True)


# ## Inferences:
# * Clients who own a car are half in number of the clients who dont own a car. But based on the percentage of deault, there is no correlation between owning a car and loan repayment as in both cases the default percentage is almost same.

# ### Checking if owning a car is related to loan repayment status

# In[ ]:



SMK1('FLAG_OWN_CAR')


# ## Inferences:
# * The clients who own real estate are more than double of the ones that don't own. But the defaulting rate of both categories are around the same (~8%).*  Thus there is no correlation between owning a reality and defaulting the loan.

# # Analyzing Housing Type based on loan repayment status

# In[113]:


SMK1("NAME_HOUSING_TYPE",True,True,True)


# ### Inferences:
# * Majority of people live in House/apartment
# * People living in office apartments have lowest default rate
# * People living with parents (~11.5%) and living in rented apartments(>12%) have higher probability of defaulting

# # Analyzing Family status based on loan repayment status

# In[114]:


SMK1("NAME_FAMILY_STATUS",False,True,True)


# ## Inferences:
# * Most of the people who have taken loan are married, followed by Single/not married and civil marriage
# * In terms of percentage of not repayment of loan, Civil marriage has the highest percent of not repayment (10%), with Widow the lowest (exception being Unknown).

# # Analyzing Education Type based on loan repayment status

# In[115]:


SMK1("NAME_EDUCATION_TYPE",True,True,True)


# ### Inferences:
# * Majority of the clients have Secondary / secondary special education, followed by clients with Higher education. Only a very * small number having an academic degree
# * The Lower secondary category, although rare, have the largest rate of not returning the loan (11%). The people with Academic degree have less than 2% defaulting rate.

# ### Analyzing Income Type based on loan repayment status

# In[116]:



SMK1("NAME_INCOME_TYPE",True,True,False)


# ### Inferences:
# * Most of applicants for loans have income type as Working, followed by Commercial associate, Pensioner and State servant.
# * The applicants with the type of income Maternity leave have almost 40% ratio of not returning loans, followed by Unemployed (37%). The rest of types of incomes are under the average of 10% for not returning loans.
# * Student and Businessmen, though less in numbers do not have any default record. Thus these two category are safest for providing loan.

# # Analyzing Region rating where applicant lives based on loan repayment status
# 

# In[117]:


SMK1("REGION_RATING_CLIENT",False,False,True)


# ## Inferences:
# * Most of the applicants are living in Region_Rating 2 place.
# * Region Rating 3 has the highest default rate (11%)
# * Applicant living in Region_Rating 1 has the lowest probability of defaulting, thus safer for approving loans

# ### Analyzing Occupation Type where applicant lives based on loan repayment status

# In[119]:


SMK1("OCCUPATION_TYPE",False,True,False)


# ## Inferences:
# * Most of the loans are taken by Laborers, followed by Sales staff. IT staff take the lowest amount of loans.
# * The category with highest percent of not repaid loans are Low-skill Laborers (above 17%), followed by Drivers and         Waiters/barmen staff, Security staff, Laborers and Cooking staff.

# # Checking Loan repayment status based on Organization type

# In[118]:


SMK1("ORGANIZATION_TYPE",True,True,False)


# ### Inferences:
# Organizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self employed people have relative high defaulting rate, and thus should be avoided to be approved for loan or provide loan with higher interest rate to mitigate the risk of defaulting.
# Most of the people application for loan are from Business Entity Type 3
# For a very high number of applications, Organization type information is unavailable(XNA)
# It can be seen that following category of organization type has lesser defaulters thus safer for providing loans:
# Trade Type 4 and 5
# Industry type 8

# # Analyzing Flag_Doc_3 submission status based on loan repayment status

# In[120]:


SMK1("FLAG_DOCUMENT_3",False,False,True)


# ### Inferences:
# *  There is no significant correlation between repayers and defaulters in terms of submitting document 3 as we see even if         applicants have submitted the document, they have defaulted a slightly more (~9%) than who have not submitted the document (6%)

# # Analyzing Amount_Credit based on loan repayment status

# In[121]:



SMK1("AMT_REQ_CREDIT_BUREAU_DAY",False,False,False)


# ### Inferences:
# * More than 80% of the loan provided are for amount less than 900,000
# * People who get loan for 300-600k tend to default more than others.

# # Analyzing Amount_Income Range based on loan repayment status
# 

# In[122]:


SMK1("AMT_REQ_CREDIT_BUREAU_YEAR",False,False,False)


# #### Analyzing Number of family members based on loan repayment status

# In[123]:


SMK1("CNT_CHILDREN",True, False, False)


# ### Inferences:
# *  children's follows the same trend as  where having more family members increases the risk of defaulting

# ####  Categorical Bi/Multivariate Analysis

# In[124]:


Madhan.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].describe()


# ##  Bifurcating the Madhan dataframe based on Target value 0 and 1 for correlation and other analysis

# In[125]:


Madhan.columns


#  #### Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

# In[126]:



cols_for_correlation = ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR']


Repayer_df = Madhan.loc[Madhan['TARGET']==0, cols_for_correlation] # Repayers
Defaulter_df = Madhan.loc[Madhan['TARGET']==1, cols_for_correlation] # Defaulters


# ### # Getting the top 10 correlation for the Repayers data

# In[127]:



corr_repayer = Repayer_df.corr()
corr_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape),k=1).astype(np.bool))
corr_df_repayer = corr_repayer.unstack().reset_index()
corr_df_repayer.columns =['VAR1','VAR2','Correlation']
corr_df_repayer.dropna(subset = ["Correlation"], inplace = True)
corr_df_repayer["Correlation"]=corr_df_repayer["Correlation"].abs() 
corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True) 
corr_df_repayer.head(10)


# In[128]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Repayer_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# #### Inferences:
# * Correlating factors amongst repayers:
# * Credit amount is highly correlated with
#   amount of goods price
#   loan annuity
#   total income
# * We can also see that repayers have high correlation in number of days employed.

# ### # Getting the top 10 correlation for the Defaulter data

# In[129]:



corr_Defaulter = Defaulter_df.corr()
corr_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape),k=1).astype(np.bool))
corr_df_Defaulter = corr_Defaulter.unstack().reset_index()
corr_df_Defaulter.columns =['VAR1','VAR2','Correlation']
corr_df_Defaulter.dropna(subset = ["Correlation"], inplace = True)
corr_df_Defaulter["Correlation"]=corr_df_Defaulter["Correlation"].abs()
corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)
corr_df_Defaulter.head(10)


# In[130]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Defaulter_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# ### Inferences:
# * Credit amount is highly correlated with amount of goods price which is same as repayers.
# * But the loan annuity correlation with credit amount has slightly reduced in defaulters(0.75) when compared to repayers(0.77)
# * We can also see that repayers have high correlation in number of days employed(0.62) when compared to defaulters(0.58).
# * There is a severe drop in the correlation between total income of the client and the credit amount(0.038) amongst defaulters    whereas it is 0.342 among repayers.
# * Days_birth and number of children correlation has reduced to 0.259 in defaulters when compared to 0.337 in repayers.
# * There is a slight increase in defaulted to observed count in social circle among defaulters(0.264) when compared to           repayers(0.254)

# In[ ]:


## # Plotting the numerical columns related to amount as distribution plot to see density


# In[131]:



amount = Madhan[[ 'CNT_CHILDREN','AMT_CREDIT','AMT_ANNUITY','SK_ID_CURR']]

fig = plt.figure(figsize=(16,12))

for i in enumerate(amount):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(Defaulter_df[i[1]], hist=False, color='r',label ="Defaulter")
    sns.distplot(Repayer_df[i[1]], hist=False, color='g', label ="Repayer")
    plt.title(i[1], fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    
plt.legend()

plt.show() 


# ## Inferences:
# * Most no of loans are given for goods price below 10 lakhs
# * Most people pay annuity below 50000 for the credit loan
# * Credit amount of the loan is mostly less then 10 lakhs
# * The repayers and defaulters distribution overlap in all the plots and hence we cannot use any of these variables in isolation to make a decision

# ###  Numerical Bivariate Analysis

# ### Checking the relationship between Goods price and credit and comparing with loan repayment staus

# In[132]:



bivariate_rel('AMT_GOODS_PRICE','AMT_CREDIT',Madhan,"TARGET", "line", ['g','r'], False,(15,6))


# ## Inferences:
# * When the credit amount goes beyond 3M, there is an increase in defaulters.

# ##### Plotting pairplot between amount variable to draw reference against loan repayment status

# In[133]:



amount = Madhan[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE','TARGET']]
amount = amount[(amount["AMT_GOODS_PRICE"].notnull()) & (amount["AMT_ANNUITY"].notnull())]
ax= sns.pairplot(amount,hue="TARGET",palette=["g","r"])
ax.fig.legend(labels=['Repayer','Defaulter'])
plt.show()


# #### Inferences:
# * When amt_annuity >15000 amt_goods_price> 3M, there is a lesser chance of defaulters
# * AMT_CREDIT and AMT_GOODS_PRICE are highly correlated as based on the scatterplot where most of the data are consolidated in form of a line
# * There are very less defaulters for AMT_CREDIT >3M
# * Inferences related to distribution plot has been already mentioned in previous distplot graphs inferences section

# ### Merged Dataframes Analysis

# ### merge both the dataframe on SK_ID_CURR with Inner Joins

# In[134]:



loan_process_df = pd.merge(Madhan, Madhan1, how='inner', on='SK_ID_CURR')
loan_process_df.head()


# ### Checking the details of the merged dataframe

# In[135]:


loan_process_df.shape


# #### Checking the element count of the dataframe

# In[136]:


loan_process_df.size


# ###  checking the columns and column types of the dataframe
# 

# In[137]:


loan_process_df.info()


# # Checking merged dataframe numerical columns statistics

# In[138]:


loan_process_df.describe()


# ###  Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

# In[139]:




L0 = loan_process_df[loan_process_df['TARGET']==0] # Repayers
L1 = loan_process_df[loan_process_df['TARGET']==1] # Defaulters


# ### Plotting Contract Status vs purpose of the loan:

# In[140]:


univariate_merged("NAME_CASH_LOAN_PURPOSE",L0,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))


# ## Inferences:
# * Loan purpose has high number of unknown values (XAP, XNA)
# * Loan taken for the purpose of Repairs seems to have highest default rate
# * A very high number application have been rejected by bank or refused by client which has purpose as repair or other. This shows that purpose repair is taken as high risk by bank and either they are rejected or bank offers very high loan interest rate which is not feasible by the clients, thus they refuse the loan.

# ### # Checking the Contract Status based on loan repayment status and whether there is any business loss or financial loss

# In[ ]:



#univariate_merged("NAME_CONTRACT_STATUS",loan_process_df,"SK_ID_CURR",['g','r'],False,(12,8))
#g = loan_process_df.groupby("NAME_CONTRACT_STATUS")["SK_ID_CURR"]
#df1 = pd.concat([g.value_counts(),round(g.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))
#df1['Percentage'] = df1['Percentage'].astype(str) +"%" # adding percentage symbol in the results for understanding
#print (df1)


# #### Inferences:
# * 90% of the previously cancelled client have actually repayed the loan. Revisiting the interest rates would increase business opoortunity for these clients
# * 88% of the clients who have been previously refused a loan has payed back the loan in current case.
# * Refual reason should be recorded for further analysis as these clients would turn into potential repaying customer.

# ### plotting the relationship between income total and contact status

# In[ ]:



merged_pointplot("NAME_CONTRACT_STATUS",'AMT_INCOME_TOTAL')


# ## Inferences:
# * The point plot show that the people who have not used offer earlier have defaulted even when there average income is higher than others

# ###  plotting the relationship between people who defaulted in last 60 days being in client's social circle and contact status

# In[ ]:



merged_pointplot("NAME_CONTRACT_STATUS",'DEF_60_CNT_SOCIAL_CIRCLE')

