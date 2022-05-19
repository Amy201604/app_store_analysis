#!/usr/bin/env python
# coding: utf-8

# ## Load Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.2f}'.format
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[3]:


sns.set_style("white")


# ## Load data

# In[4]:


account = pd.read_csv('account_dat.csv', parse_dates=['create_dt'])
app = pd.read_csv('app_dat.csv')
category = pd.read_csv('category_ref.csv')
device = pd.read_csv('device_ref.csv')
inapp = pd.read_csv('in-app_dat.csv')
transaction = pd.read_csv('transaction_dat.csv', parse_dates=['create_dt'])


# # 1 Data Processing

# ## 1.1 Explore Data

# ### Transaction table

# In[5]:


transaction.info()


# In[6]:


transaction = transaction.rename(columns = {'content_id':'transaction_content_id','device_id':'transaction_device_id'})


# In[7]:


transaction.isna().sum()


# In[8]:


transaction = transaction.rename(columns = {'create_dt':'transaction_dt'})


# In[9]:


print("Analysis period starts from ", transaction.transaction_dt.max().date()," ends by ",transaction.transaction_dt.min().date())


# In[10]:


# A ratio can be added to calculate the annual metrics 
ratio = (transaction['transaction_dt'].max() - transaction['transaction_dt'].min())/np.timedelta64(1,'D')/365


# In[11]:


missing_days = (transaction.transaction_dt.max()- transaction.transaction_dt.min())/np.timedelta64(1,'D') + 1 - transaction.transaction_dt.nunique()
print("There are {} missing days in the data".format(missing_days))


# In[12]:


len(transaction) - len(transaction.drop_duplicates())


# In[13]:


print("There are {} duplicates transaction in the data".format(len(transaction) - len(transaction.drop_duplicates())))


# In[14]:


duplicates_transaction_rate = round((len(transaction) - len(transaction.drop_duplicates()))/ len(transaction) * 100,2)


# In[15]:


print("There is {} % duplicates transaction in the data".format(duplicates_transaction_rate))


# #### The duplicates transaction means the same account download/purchase the same content on the same same device on the same day at the same price more than once. It should be very unlikely to happen. So I will drop the duplicates.

# #### Since there is no transaction_id in the data, I cannot be 100% sure if theose the same transactions.

# In[16]:


transaction_de_dup = transaction.drop_duplicates()


# ### Account table

# In[17]:


account.info()


# In[18]:


account = account.rename(columns = {'create_dt':'account_create_dt'})


# In[19]:


account.head()


# In[20]:


len(account) == len(account.drop_duplicates())


# In[21]:


account.isna().sum()


# ### App table

# In[22]:


app.info()


# In[23]:


app = app.rename(columns = {'content_id':'parent_app_content_id'})


# In[24]:


len(app) == len(app.drop_duplicates())


# In[25]:


app.isna().sum()


# In[26]:


app.parent_app_content_id.nunique()


# In[27]:


app.app_name.nunique()


# In[28]:


app_duplicates = app.groupby(['app_name']).agg(app_ct =('parent_app_content_id', 'nunique'))
app_duplicates[app_duplicates['app_ct'] >1]


# ##### There are 4 duplicates caused by unknown App names. Column 'parent_app_content_id' should be used as unique app identifier. 

# ### Inapp Table

# In[29]:


inapp.info()


# In[30]:


inapp = inapp.rename(columns = {'type':'inapp_content_type'})


# In[31]:


len(inapp) == len(inapp.drop_duplicates())


# In[32]:


inapp.isna().sum()


# In[33]:


inapp.parent_app_content_id.nunique()


# In[34]:


inapp.content_id.nunique()


# In[35]:


inapp.inapp_content_type.nunique()


# In[36]:


inapp.inapp_content_type.unique()


# #### There are fewer parent_app_content_id in Inapp table comapred to that in App table. In that case, there would be content cannot join with columns in the App table. 

# ### Device Table

# In[37]:


device.info()


# In[38]:


len(device) == len(device.drop_duplicates())


# In[39]:


device.isna().sum()


# In[40]:


device.device_name.nunique()


# In[41]:


device.device_name.unique()


# ### Category Table

# In[42]:


category.info()


# In[43]:


len(category) == len(category.drop_duplicates())


# In[44]:


category.isna().sum()


# In[45]:


category.category_name.nunique()


# In[46]:


category.category_name.unique()


# ## 1.2 Join App with inApp, app device table and category table

# In[47]:


app.head()


# In[48]:


inapp.head()


# In[49]:


app_inapp = app.merge(inapp, on = 'parent_app_content_id', how = 'outer')
# Using outer join in case there are parent_app_content_id in one table but not the other table.


# In[50]:


app_inapp.isnull().sum()


# In[51]:


app_inapp[app_inapp['content_id'].isna()].head()


# In[52]:


app_df = app_inapp.merge(category, on = 'category_id', how = 'left')                .merge(device, on = 'device_id', how = 'left')


# In[53]:


app_df = app_df.rename(columns = {'device_id':'app_device_id','device_name':'app_device_name'})


# In[54]:


app_df.info()


# In[55]:


app_df.head()


# In[56]:


app_df.isna().sum()


# In[57]:


app_df[app_df['content_id'].isna()].head()


# In[58]:


app_df[app_df['content_id'].isna()]['parent_app_content_id'].nunique()


# In[59]:


## Since content_id would be the join key later on, it can not be NaN. Some transaction_contetn_id is the parent_app_contetn_id based on data exploartion. I added a new column for those has no contetn_id. 
app_df['content_id_modified'] = app_df['content_id'].fillna(app_df['parent_app_content_id'])


# In[60]:


app_df[app_df['content_id'].isna()].head()


# In[61]:


app_df.isna().sum()


# ## 1.3 Clean up App name

# In[62]:


app_df['app_name'] = app_df['app_name'].str.replace('(^\s+|\;|\-|\,|\:|\)|\()', '', regex=True).str.strip().str.title()


# ## 1.4 Join transaction table with App info

# In[63]:


transaction_de_dup.info()


# In[64]:


df = transaction_de_dup.merge(app_df, left_on = 'transaction_content_id', right_on = 'content_id_modified', how = 'left')
# using 'content_id_modified' as the join key to avoid joining on NaN


# In[65]:


len(transaction_de_dup) == len(df)


# There is no duplcates coming from the join.

# In[66]:


df.isna().sum()


# In[67]:


df[df['parent_app_content_id'].isna()].head()


# Some content_id cannot be joined by column content_id, but can be joined by parent_app_content_id.

# In[68]:


df_fix = df[df['parent_app_content_id'].isna()][['transaction_dt', 'transaction_content_id', 'acct_id', 'price','transaction_device_id']]                .merge(app_df[['app_name', 'parent_app_content_id', 'category_id','category_name', 'app_device_id', 'app_device_name']], left_on = 'transaction_content_id', right_on = 'parent_app_content_id', how = 'left')


# In[69]:


df_fix.isna().sum()


# In[70]:


df_fix.head()


# In[71]:


len(df_fix)
# There are duplciates since 1 app can have multiple content_ids. 


# In[72]:


len(df_fix.drop_duplicates()) == len(df[df['parent_app_content_id'].isna()])


# In[73]:


len(df[~df['parent_app_content_id'].isna()]) + len(df_fix.drop_duplicates()) == len(transaction_de_dup)


# There is no duplicates after de-dup.

# In[74]:


# Union the two parts together
df_txn_app = pd.concat([df[~df['parent_app_content_id'].isna()][['transaction_dt', 'transaction_content_id', 'acct_id', 'price',
       'transaction_device_id', 'app_name', 'parent_app_content_id','category_id','category_name', 'app_device_id', 'app_device_name']], df_fix.drop_duplicates()], ignore_index=True)


# In[75]:


len(df_txn_app) == len(transaction_de_dup)


# In[76]:


df_txn_app.head()


# In[77]:


df_txn_app.isna().sum()


# In[78]:


df_txn_app_final = df_txn_app.merge(app_df[['content_id','inapp_content_type']], left_on = 'transaction_content_id', right_on = 'content_id', how = 'left')


# In[79]:


df_txn_app_final.head()


# In[80]:


len(df_txn_app_final)  == len(transaction_de_dup)


# In[81]:


df_txn_app_final.isna().sum()


# ## 1.5 Join transaction table with device and account table

# In[82]:


device.head()


# In[83]:


transaction_df = df_txn_app_final.merge(device, left_on = 'transaction_device_id', right_on = 'device_id', how = 'left')                                 .merge(account, on = 'acct_id', how = 'left')
transaction_df = transaction_df.rename(columns = {'device_name':'transaction_device_name'})
transaction_df = transaction_df.drop(['device_id'], axis = 1)


# In[84]:


transaction_df.head()


# In[85]:


transaction_df.isna().sum()


# In[86]:


transaction_df[transaction_df['account_create_dt'].isna()].head()


# In[87]:


transaction_df[transaction_df['account_create_dt'].isna()]['acct_id'].nunique()


# In[88]:


len(transaction_df) == len(transaction_de_dup)


# In[89]:


transaction_df.info()


# In[90]:


transaction_df['transaction_device_id'] = transaction_df['transaction_device_id'].astype('object')
transaction_df['app_device_id'] = transaction_df['app_device_id'].astype('object')


# In[91]:


transaction_df.describe()


# ## 1.6 Detect outliers

# ### Account level summery

# In[92]:


spend_per_account = transaction_df.pivot_table(index = ['acct_id'], columns = ['category_name'], values = ['price'], aggfunc = sum).fillna(0)


# In[93]:


spend_per_account.columns = spend_per_account.columns.droplevel(0)
spend_per_account = spend_per_account.reset_index()
spend_per_account.columns = ['acct_id', 'Entertainment_revenue', 'Games_revenue', 'Photos & Videos_revenue','Social Networking_revenue', 'Utilities_revenue']


# In[94]:


txn_per_account = transaction_df.groupby(['acct_id']).agg(total_revenue = ('price',sum))
txn_per_account['total_txn'] = transaction_df.groupby(['acct_id']).size()
txn_per_account['paid_acct_flag'] = np.where(txn_per_account['total_revenue']>0 ,'Purchased Paid App', 'Free Only')
txn_per_account = txn_per_account.reset_index()


# In[95]:


account_summary = account.merge(spend_per_account, on ='acct_id', how = 'left')                    .merge(txn_per_account, on = 'acct_id', how = 'left')
account_summary = account_summary.fillna(0)
account_summary['no_txn_flag'] = np.where(account_summary['total_txn'] == 0, 'Dormant', 'Active')
account_summary['paid_acct_flag'] = account_summary['paid_acct_flag'].replace(0,'Dormant')


# In[96]:


account_summary['acct_tenure'] = transaction['transaction_dt'].max() - account_summary['account_create_dt']
account_summary['acct_tenure_days'] = account_summary['acct_tenure']/np.timedelta64(1,'D')
account_summary['acct_tenure_months'] = account_summary['acct_tenure']/np.timedelta64(1,'M')
account_summary['acct_tenure_years'] = account_summary['acct_tenure']/np.timedelta64(1,'Y')


# In[97]:


tenure_cut_labels = ['no account','1 month', '2- 6 months', '6 months - 1 year', '1- 3 years','3 - 5 years', '5 years+']
tenure_cut_bins = [account_summary['acct_tenure_months'].min(), 0, 1, 6, 12,36, 60, account_summary['acct_tenure_months'].max()]
account_summary['Tenure_Band'] = pd.cut(account_summary['acct_tenure_months'], bins = tenure_cut_bins, labels = tenure_cut_labels)


# In[98]:


account_summary.info()


# In[99]:


account_summary.head()


# In[100]:


len(account_summary) == len(account)


# In[101]:


account_summary.describe()


# In[102]:


account_summary[(account_summary['total_revenue'] == account_summary['total_revenue'].max()) | (account_summary['total_txn'] == account_summary['total_txn'].max())]


# There are extreme values (outliers).

# In[103]:


txn_acct = account_summary.groupby(['no_txn_flag']).agg(no_txn_acct_ct = ('acct_id', 'nunique'))


# In[104]:


fig1, ax1 = plt.subplots()
ax1.pie(txn_acct.no_txn_acct_ct, explode= (0, 0.1) , labels= txn_acct.index, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['blue','grey'], textprops={'color':"w",'size':16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Active Account Proportion', fontsize=24)
plt.legend(loc="best")
plt.show()


# ### Avg spend per account to detect outliers

# In[105]:


paid = account_summary.groupby(['paid_acct_flag']).agg(paid_acct_ct = ('acct_id', 'nunique'))


# In[106]:


paid


# In[107]:


paid.to_csv('paid.csv')


# In[108]:


fig1, ax1 = plt.subplots()
ax1.pie(paid.paid_acct_ct, explode= (0, 0, 0.1) , labels= paid.index, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['blue','brown','green'], textprops={'color':"w",'size':16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Paid Account Proportion', fontsize=24)
plt.legend(loc="best")
plt.show()


# In[109]:


sns.displot(account_summary, x='total_revenue')


# In[110]:


sns.displot(account_summary[account_summary['total_revenue'] != 0], x='total_revenue')


# In[111]:


sns.displot(account_summary[(account_summary['total_revenue'] != 0) & (account_summary['total_revenue'] <= 2000)], x='total_revenue')


# In[112]:


sns.displot(account_summary[(account_summary['total_revenue'] != 0) & (account_summary['total_revenue'] <= 10000)], x='total_revenue')


# #### Accounts with more than 10,000 spend in the 110 days are considered as outliers, and will be excluded from this analysis. 

# In[113]:


outliers1 = account_summary[account_summary['total_revenue'] > 10000]['acct_id']


# In[114]:


print("In total, there are {} outliers caused by spend amount.".format(outliers1.nunique()))


# ### Avg number of transactions per account to detect outliers

# In[115]:


sns.displot(account_summary, x='total_txn')


# In[116]:


sns.displot(account_summary[account_summary['total_txn'] != 0], x='total_txn')


# In[117]:


sns.displot(account_summary[(account_summary['total_txn'] != 0) & (account_summary['total_txn'] <= 1000)], x='total_txn')


# In[118]:


sns.displot(account_summary[(account_summary['total_txn'] != 0) & (account_summary['total_txn'] <= 400)], x='total_txn')


# #### Accounts with more than 1,000 transactions in the 110 days are considered as outliers, and will be excluded from this analysis. 

# In[119]:


outliers2 = account_summary[account_summary['total_txn'] > 1000]['acct_id']


# In[120]:


print("In total, there are {} outliers caused by transaction number.".format(outliers2.nunique()))


# In[121]:


account_summary[(account_summary['total_txn'] <= 1000) & (account_summary['total_revenue'] <= 10000)].describe()


# In[122]:


account_summary[(account_summary['total_txn'] <= 1000) & (account_summary['total_revenue'] <= 10000) & (account_summary['total_txn'] != 0)].describe()


# #### The account level transaction distribution looks like a bimodel distribution, indicating two different groups

# In[123]:


account_summary['acct_txn_group'] = np.where(account_summary['total_txn'] <= 101, 'Low','High')
account_summary['acct_txn_group'] = np.where(account_summary['paid_acct_flag'] == 'Dormant', 'Dormant', account_summary['acct_txn_group'] )


# In[124]:


account_summary.head()


# In[125]:


account_summary.to_csv('account_summary.csv')


# In[126]:


transaction_df = transaction_df.merge(account_summary[['acct_id','acct_txn_group']], on = 'acct_id', how = 'left')


# ### The relationsip between payment type and tenure

# In[127]:


account_summary.groupby(['Tenure_Band','payment_type']).size()


# In[128]:


ax = sns.countplot(x = 'payment_type', hue = 'Tenure_Band', data = account_summary)
ax.set_title('Number of Account by Payment Type', fontsize=20)
ax.set_xlabel("Payment Type", fontsize = 16)
ax.set_ylabel("Number of Account", fontsize = 16)


# ### The relationship between account active status with payment type

# In[129]:


account_summary.groupby(['payment_type','no_txn_flag']).size()


# ### The corelation between account active status and payment type cannot be proved based on the data

# ### Analyze the two 'acct_txn_group'

# In[130]:


account_summary.groupby(['Tenure_Band','acct_txn_group']).size()


# #### Longer tenured accounts are more likely to be highly active. Shorter tenure accounts are more likely to be dormant or low active. 

# In[131]:


ax = sns.countplot(x = 'acct_txn_group', hue = 'Tenure_Band', data = account_summary)
ax.set_title('Number of Account by Account Group', fontsize=20)
ax.set_xlabel("Account Group", fontsize = 16)
ax.set_ylabel("Number of Account", fontsize = 16)


# In[132]:


account_summary.groupby(['acct_txn_group'])['acct_tenure_years'].mean()


# #### Highly active accounts are more likely to have long tenure.

# In[133]:


ax = sns.countplot(x = 'acct_txn_group', hue = 'payment_type', data = account_summary)
ax.set_title('Number of Account by Payment Type', fontsize=20)
ax.set_xlabel("Account Group", fontsize = 16)
ax.set_ylabel("Number of Account", fontsize = 16)


# ## 1.7 Clean data and add meaningful columns

# In[134]:


transaction_df_final = transaction_df[(~transaction_df['acct_id'].isin(outliers1)) & (~transaction_df['acct_id'].isin(outliers2))]


# In[135]:


transaction_df_final['transaction_year_month'] = transaction_df_final['transaction_dt'].dt.to_period('M')
transaction_df_final['transaction_year'] = transaction_df_final['transaction_dt'].dt.year
transaction_df_final['transaction_month'] = transaction_df_final['transaction_dt'].dt.month
transaction_df_final['transaction_weekday'] = transaction_df_final['transaction_dt'].dt.weekday
transaction_df_final['transaction_weekday_nm'] = transaction_df_final['transaction_dt'].dt.day_name()


# In[136]:


transaction_df_final['acct_tenure'] = transaction_df_final['transaction_dt'] - transaction_df_final['account_create_dt']
transaction_df_final['acct_tenure_days'] = transaction_df_final['acct_tenure']/np.timedelta64(1,'D')
transaction_df_final['acct_tenure_months'] = transaction_df_final['acct_tenure']/np.timedelta64(1,'M')
transaction_df_final['acct_tenure_years'] = transaction_df_final['acct_tenure']/np.timedelta64(1,'Y')


# In[137]:


transaction_df_final['transaction_without_acct'] = np.where(transaction_df_final['transaction_dt'] < transaction_df_final['account_create_dt'],1,0)


# In[138]:


tenure_cut_labels = ['no account','1 month', '2- 6 months', '6 months - 1 year', '1- 3 years','3 - 5 years', '5 years+']
tenure_cut_bins = [transaction_df_final['acct_tenure_months'].min(), 0, 1, 6, 12,36, 60, transaction_df_final['acct_tenure_months'].max()]
transaction_df_final['Tenure_Band'] = pd.cut(transaction_df_final['acct_tenure_months'], bins = tenure_cut_bins, labels = tenure_cut_labels)


# In[139]:


transaction_df_final['transaction_week'] = transaction_df_final['transaction_dt'].dt.week
transaction_df_final["First_day_of_the_week"] = transaction_df_final['transaction_dt'] - transaction_df_final['transaction_weekday'] * np.timedelta64(1, 'D')


# In[140]:


transaction_df_final['free_transaction_flag'] = np.where(transaction_df_final['price'] == 0, 1,0)


# In[141]:


price_cut_labels = ['free','below $ 5', '$ 5-10', '$ 10-20', '$20+']
price_cut_bins = [0, 0.0001, 5, 10,20, transaction_df_final['price'].max()]
transaction_df_final['Price_Band'] = pd.cut(transaction_df_final['price'], bins = price_cut_bins, labels = price_cut_labels)
transaction_df_final['Paid_acct_Flag'] = np.where(transaction_df_final.groupby(['acct_id'])['price'].transform(sum)>0 ,1 ,0)
transaction_df_final['Price_Band'] = transaction_df_final['Price_Band'].fillna('free')


# In[142]:


transaction_df_final.sort_values(by= 'acct_id').head()


# In[143]:


transaction_df_final.info()


# In[144]:


transaction_df_final.describe()


# ## 1.8 Check high level transaction data distribution

# In[145]:


transaction_df_final.groupby(['transaction_year_month']).size()


# In[146]:


ax = sns.countplot(x = 'transaction_year_month', data = transaction_df_final, order = transaction_df_final['transaction_year_month'].value_counts().index)
ax.bar_label(container=ax.containers[0], labels = transaction_df_final['transaction_year_month'].value_counts().values)
ax.set_title('Number of Transaction by Time', fontsize=20)
ax.set_xlabel("Year - Month", fontsize = 16)
ax.set_ylabel("Number of Transaction", fontsize = 16)


# In[147]:


ax = sns.countplot(x = transaction_df_final['Price_Band'], order = transaction_df_final['Price_Band'].value_counts().index)
ax.bar_label(container=ax.containers[0], labels=transaction_df_final['Price_Band'].value_counts().values)
ax.set_title('Number of Transaction by Price', fontsize=20)
ax.set_xlabel("Price", fontsize = 16)
ax.set_ylabel("Number of Transaction", fontsize = 16)


# In[148]:


ax = sns.countplot(x = 'Tenure_Band', data = transaction_df_final, order = transaction_df_final['Tenure_Band'].value_counts(ascending=False).index)
ax.bar_label(container=ax.containers[0], labels=transaction_df_final['Tenure_Band'].value_counts().values)
ax.set_title('Number of Transaction by Tenure', fontsize=20)
ax.set_xlabel("Account Tenure", fontsize = 16)
ax.set_ylabel("Number of Transaction", fontsize = 16)


# In[149]:


ax = sns.countplot(x = 'Tenure_Band', data = transaction_df_final[transaction_df_final['free_transaction_flag'] != 1], order = transaction_df_final['Tenure_Band'].value_counts(ascending=False).index)
ax.set_title('Number of Paid Transaction by Tenure', fontsize=20)
ax.set_xlabel("Account Tenure", fontsize = 16)
ax.set_ylabel("Number of Transaction", fontsize = 16)


# In[150]:


transaction_df_final[transaction_df_final['free_transaction_flag'] != 1].groupby(['Tenure_Band']).size()


# In[151]:


tenure_txn = transaction_df_final.groupby(['Tenure_Band']).agg(acct_ct= ('acct_id','nunique'))
tenure_txn['txn'] = transaction_df_final.groupby(['Tenure_Band']).size()
tenure_txn['avg_txn'] = tenure_txn['txn']/tenure_txn['acct_ct']


# In[152]:


tenure_txn


# In[153]:


sns.barplot(x = tenure_txn.index, y = 'avg_txn', data = tenure_txn)


# In[154]:


tenure_txn.to_csv('tenure_txn.csv')


# In[155]:


acct_txn_group_summary = transaction_df_final.groupby(['acct_txn_group']).agg(acct_ct = ('acct_id','nunique'), total_revenue = ('price',sum))
acct_txn_group_summary['total_txn'] = transaction_df_final.groupby(['acct_txn_group']).size()
acct_txn_group_summary['avg_revenue'] = acct_txn_group_summary['total_revenue']/ acct_txn_group_summary['acct_ct']
acct_txn_group_summary['avg_txn'] = acct_txn_group_summary['total_txn']/ acct_txn_group_summary['acct_ct']


# In[156]:


acct_txn_group_summary


# #### All the paid accounts are high active accounts.  Not sure if that comes from the selection bias.

# # 2 Performance Dashboard

# ## 2.1 Daily Trends

# ### Daily Performance Trend

# In[157]:


performance_daily = transaction_df_final.groupby(['transaction_dt']).agg(active_acct_ct = ('acct_id', 'nunique'),                                                                    active_unique_app = ('parent_app_content_id', 'nunique'),                                                                    dalily_sales = ('price', sum),                                                                    free_transaction_ct = ('free_transaction_flag',sum)
                                                                   )
performance_daily['transaction_ct'] = transaction_df_final.groupby(['transaction_dt']).size()


# In[158]:


performance_daily['paid_transaction_ct'] = performance_daily['transaction_ct'] - performance_daily['free_transaction_ct']
performance_daily['avg_free_transaction_ct_per_acct'] = performance_daily['free_transaction_ct']/ performance_daily['active_acct_ct']
performance_daily['avg_paid_transaction_ct_per_acct'] = performance_daily['paid_transaction_ct']/ performance_daily['active_acct_ct']
performance_daily['avg_transaction_ct_per_acct'] = performance_daily['transaction_ct']/ performance_daily['active_acct_ct']
performance_daily['avg_sales_per_acct'] = performance_daily['dalily_sales']/ performance_daily['active_acct_ct']


# In[159]:


performance_daily.head()


# In[160]:


performance_daily.to_csv('performance_daily.csv')


# In[161]:


ax1 = sns.lineplot(x = 'transaction_dt', y = 'dalily_sales', data = performance_daily, color = 'orange')
ax2 = ax1.twinx()
ax2 = sns.lineplot(x = 'transaction_dt', y = 'transaction_ct', data = performance_daily, color = 'purple')

ax1.set_title('Daily Sales Trends', fontsize=20)
ax1.set_xlabel("Date", fontsize = 16)
ax1.set_ylabel("Daily Sales Amount", fontsize = 16)
ax1.set(ylim=(0, None))
# ax1.yaxis.set_major_formatter(plt.FuncFormatter('$ {}'.format))

ax2.set_ylabel("Daily Number of Transactions", fontsize = 16)
ax2.set(ylim=(0, None))


# In[162]:


ax1 = sns.lineplot(x = 'transaction_dt', y = 'active_acct_ct', data = performance_daily, color = 'brown')

ax1.set_title('Daily Active Account Trends', fontsize=20)
ax1.set_xlabel("Date", fontsize = 16)
ax1.set_ylabel("Daily Active Account", fontsize = 16)
ax1.set(ylim=(0, 25000))


# In[163]:


f, ax = plt.subplots(1, 1)
ax.plot_date(performance_daily.index, performance_daily['avg_free_transaction_ct_per_acct'], color="blue", label="Average Number of Free Transactions", linestyle="-")
ax.plot_date(performance_daily.index, performance_daily['avg_paid_transaction_ct_per_acct'], color="brown", label='Average Number of Paid Transactions', linestyle="-")
ax.plot_date(performance_daily.index, performance_daily['avg_transaction_ct_per_acct'], color="black", label='Average Number of Total Transactions', linestyle="-")

ax.legend()
ax.set_title('Daily Transaction Trends', fontsize=20)
ax.set_xlabel("Date", fontsize = 16)
ax.set_ylabel("Number of Transactions", fontsize = 16)

plt.gcf().autofmt_xdate()
plt.show()


# ### Increas on free content transaction

# ### Daily Performance Trend

# In[164]:


performance_daily_category = transaction_df_final.groupby(['transaction_dt','category_name']).agg(active_acct_ct = ('acct_id', 'nunique'),                                                                    dalily_sales = ('price', sum),                                                                    free_transaction_ct = ('free_transaction_flag',sum)
                                                                   )
performance_daily_category['transaction_ct'] = transaction_df_final.groupby(['transaction_dt','category_name']).size()


# In[165]:


performance_daily_category['paid_transaction_ct'] = performance_daily_category['transaction_ct'] - performance_daily_category['free_transaction_ct']
performance_daily_category['avg_free_transaction_ct_per_acct'] = performance_daily_category['free_transaction_ct']/ performance_daily_category['active_acct_ct']
performance_daily_category['avg_paid_transaction_ct_per_acct'] = performance_daily_category['paid_transaction_ct']/ performance_daily_category['active_acct_ct']
performance_daily_category['avg_transaction_ct_per_acct'] = performance_daily_category['transaction_ct']/ performance_daily_category['active_acct_ct']
performance_daily_category['avg_sales_per_acct'] = performance_daily_category['dalily_sales']/ performance_daily_category['active_acct_ct']


# In[166]:


performance_daily_category.head()


# In[167]:


performance_daily_category = performance_daily_category.reset_index()


# In[168]:


performance_daily_category.to_csv('performance_daily_category.csv')


# In[169]:


ax = sns.lineplot(x = 'transaction_dt', y = 'avg_free_transaction_ct_per_acct', hue = 'category_name', data = performance_daily_category, color = 'orange')

ax.set_title('Average Free Transaction Trends', fontsize=20)
ax.set_xlabel("Date", fontsize = 16)
ax.set_ylabel("Average Free Transaction Per Active Account", fontsize = 16)
ax.set(ylim=(0, 1.4))


# In[170]:


ax = sns.lineplot(x = 'transaction_dt', y = 'avg_paid_transaction_ct_per_acct', hue = 'category_name', data = performance_daily_category, color = 'orange')

ax.set_title('Average Paid Transaction Trends', fontsize=20)
ax.set_xlabel("Date", fontsize = 16)
ax.set_ylabel("Average Paid Transaction Per Active Account", fontsize = 16)
ax.set(ylim=(0, 1.4))


# ### Spike on July 4th

# #### - The sales on July 4th, 2016 increased dramatically. It is posssibile there was a promotion or new product launch on that day. 
# #### - There was an increasing download of free Apps in August and September.
# #### - The sales and transactions have obvious weekly patterns. 

# In[171]:


july04 = transaction_df_final[transaction_df_final['transaction_dt']== '2016-07-04'].groupby(['parent_app_content_id','app_name']).agg(active_acct = ('acct_id','nunique'), revenue_by_app = ('price',sum))
july04['txn_by_app'] = transaction_df_final[transaction_df_final['transaction_dt']== '2016-07-04'].groupby(['app_name']).size()


# In[172]:


july04.sort_values(by=['revenue_by_app','txn_by_app'], ascending = [False, False]).head()


# ### Daily App Performance

# In[173]:


daily_app_performance = transaction_df_final.groupby(['parent_app_content_id','app_name','transaction_dt']).agg(active_acct = ('acct_id','nunique'), revenue_by_app = ('price',sum))
daily_app_performance['txn_by_app'] = transaction_df_final.groupby(['parent_app_content_id','app_name','transaction_dt']).size()
daily_app_performance['yesterday_txn_by_app'] = daily_app_performance.groupby(['parent_app_content_id','app_name'])['txn_by_app'].transform('shift')
daily_app_performance['app_txn_change_rate'] = daily_app_performance['txn_by_app']/ daily_app_performance['yesterday_txn_by_app'] - 1
daily_app_performance['app_txn_change'] = daily_app_performance['txn_by_app'] - daily_app_performance['yesterday_txn_by_app']


# In[174]:


daily_app_performance['yesterday_revenue_by_app'] = daily_app_performance.groupby(['parent_app_content_id','app_name'])['revenue_by_app'].transform('shift')
daily_app_performance['app_revenue_change_rate'] = daily_app_performance['revenue_by_app']/ daily_app_performance['yesterday_revenue_by_app'] - 1
daily_app_performance['app_revenue_change'] = daily_app_performance['revenue_by_app'] - daily_app_performance['yesterday_revenue_by_app']


# In[175]:


daily_app_performance = daily_app_performance.reset_index()


# In[176]:


daily_app_performance.head()


# In[177]:


daily_app_performance[daily_app_performance['transaction_dt'] == '2016-07-04'].sort_values(by= ['app_txn_change','app_txn_change_rate'], ascending = [False, False]).head(10)


# In[178]:


daily_app_performance[daily_app_performance['transaction_dt'] == '2016-07-04'].sort_values(by= ['app_revenue_change','app_revenue_change_rate'], ascending = [False, False]).head(10)


# ### The daily trend chart has obvious week patterns

# ## 2.2 Day of week effect

# In[179]:


performance_weekday = transaction_df_final.groupby(['transaction_weekday', 'transaction_weekday_nm']).agg(total_sales = ('price', sum),                                                                     purchased_acct_ct = ('acct_id', 'nunique'),                                                                     purchased_unique_app = ('parent_app_content_id', 'nunique'),                                                                    active_days = ('transaction_dt','nunique'))
performance_weekday['total_transaction_ct'] = transaction_df_final.groupby(['transaction_weekday','transaction_weekday_nm']).size()


# In[180]:


performance_weekday = performance_weekday.reset_index().sort_values(by = 'transaction_weekday')


# In[181]:


performance_weekday['avg_daily_transaction_ct'] = performance_weekday['total_transaction_ct']/performance_weekday['active_days']
performance_weekday['avg_daily_sales'] = performance_weekday['total_sales']/performance_weekday['active_days']


# In[182]:


performance_weekday


# In[183]:


performance_weekday.to_csv('performance_weekday_overall.csv')


# In[184]:


ax = sns.barplot(x='transaction_weekday_nm', y= 'avg_daily_sales', data=performance_weekday, color='grey')
ax.set_title('Daily Sales Trends', fontsize=20)
ax.set_xlabel("Weekday", fontsize = 16)
ax.set_ylabel("Daily Sales Amount", fontsize = 16)
ax.yaxis.set_major_formatter('$ {x}')
for p in ax.patches:
    ax.annotate("$ " + format(round(p.get_height()/1000,2), '.2f')+"K", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# In[185]:


ax = sns.barplot(x='transaction_weekday_nm', y= 'avg_daily_transaction_ct', data=performance_weekday, color='grey')
ax.set_title('Daily Transaction Count Trends', fontsize=20)
ax.set_xlabel("Weekday", fontsize = 16)
ax.set_ylabel("Daily Transaction Count", fontsize = 16)
for p in ax.patches:
    ax.annotate(format(round(p.get_height()/1000,2), '.2f')+"K", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# #### Poeple like to download and purchase App during weekend. 

# ## 2.3 Performance of each week

# ### Account Active/purchase rate of last full week

# In[186]:


account_act_week = transaction_df_final[transaction_df_final['First_day_of_the_week']== '2016-09-12'].groupby(['First_day_of_the_week','acct_id']).agg(total_revenue = ('price',sum))
account_act_week['total_txn'] = transaction_df_final.groupby(['acct_id']).size()
account_act_week['paid_acct_flag'] = np.where(account_act_week['total_revenue']>0 ,'Purchased Paid App', 'Free Only')


# In[187]:


account_act_week = account_act_week.reset_index()


# In[188]:


account_act_week.head()


# In[189]:


account_act_week.info()


# In[190]:


account_week_summary = account.merge(account_act_week, on ='acct_id', how = 'left')
account_week_summary = account_week_summary.fillna(0)
account_week_summary['no_txn_flag'] = np.where(account_week_summary['total_txn'] == 0, 'Dormant', 'Active')
account_week_summary['paid_acct_flag'] = account_week_summary['paid_acct_flag'].replace(0,'Dormant')


# In[191]:


account_week_summary.tail()


# In[192]:


account_week_summary.info()


# In[193]:


last_full_week = account_week_summary.groupby(['paid_acct_flag']).agg(paid_acct_ct = ('acct_id', 'nunique'))


# In[194]:


last_full_week 


# In[195]:


fig1, ax1 = plt.subplots()
ax1.pie(last_full_week.paid_acct_ct, explode= (0, 0,0.1) , labels= last_full_week.index, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['blue','brown','green'], textprops={'color':"w",'size':16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Paid Account Proportion', fontsize=24)
plt.legend(loc="best")
plt.show()


# ### Sales & Transacions of each week

# In[196]:


performance_week = transaction_df_final.groupby(['First_day_of_the_week']).agg(weekly_sales = ('price', sum), active_acct_ct = ('acct_id', 'nunique'), active_days = ('transaction_dt', 'nunique'))
performance_week['weekly_transaction_ct'] = transaction_df_final.groupby(['First_day_of_the_week']).size()
performance_week['avg_daily_transaction_ct'] = performance_week['weekly_transaction_ct']/performance_week['active_days']
performance_week['avg_daily_sales'] = performance_week['weekly_sales']/performance_week['active_days']


# In[197]:


performance_week['previous_week_sales'] = performance_week['weekly_sales'].shift()
performance_week['previous_week_transaction_ct'] = performance_week['weekly_transaction_ct'].shift()
performance_week['weekly_sales_increase_rate'] = performance_week['weekly_sales'] /performance_week['previous_week_sales'] - 1
performance_week['weekly_transaction_ct_increase_rate'] = performance_week['weekly_transaction_ct'] /performance_week['previous_week_transaction_ct'] - 1


# In[198]:


performance_week


# In[199]:


performance_week.to_csv('performance_week.csv')


# #### Weekly sales and transactions are quite stable over the analysis period (two weeks have missing days).

# In[200]:


ax1 = sns.lineplot(x = 'First_day_of_the_week', y = 'weekly_transaction_ct', data = performance_week)
ax1.set_title('Weekly Transaction & Sales Trends', fontsize=20)
ax1.set_xlabel("Date", fontsize = 16)
# ax1.yaxis.set_major_formatter('$ {x}')
ax1.set_ylabel("Weekly Number of Transactions", fontsize = 16)
ax1.set(ylim=(0, None))

ax2 = ax1.twinx()
ax2 = sns.lineplot(x = 'First_day_of_the_week', y = 'weekly_sales', data = performance_week, color = 'orange')
ax2.set_ylabel("Weekly Sales Amount", fontsize = 16)
ax2.set(ylim=(0, None))


# #### Dwonward trend on week 2016-08-22 was caused by the 3 missing days. 

# In[201]:


ax1 = sns.lineplot(x = 'First_day_of_the_week', y = 'avg_daily_transaction_ct', data = performance_week, color = 'blue')
ax1.set_title('Average Daily Transaction & Sales Trends', fontsize=20)
ax1.set_xlabel("Date", fontsize = 16)
ax1.set_ylabel("Average Daily Number of Transactions", fontsize = 16)
ax1.set(ylim=(0, 50000))

ax2 = ax1.twinx()
ax2 = sns.lineplot(x = 'First_day_of_the_week', y = 'avg_daily_sales', data = performance_week, color = 'orange')
ax2.set_ylabel("Average Daikly Sales Amount", fontsize = 16)
ax2.set(ylim=(0, 200000))


# #### Dwonward trend on week 2016-08-22 was caused by the 3 missing days. 

# ### Performance of each week by category

# In[202]:


performance_week_category = transaction_df_final.groupby(['First_day_of_the_week','category_name']).agg(weekly_sales = ('price', sum), active_acct_ct = ('acct_id', 'nunique'), active_days = ('transaction_dt', 'nunique'))
performance_week_category['weekly_transaction_ct'] = transaction_df_final.groupby(['First_day_of_the_week','category_name']).size()
performance_week_category['avg_daily_transaction_ct'] = performance_week_category['weekly_transaction_ct']/performance_week_category['active_days']
performance_week_category['avg_daily_sales'] = performance_week_category['weekly_sales']/performance_week_category['active_days']


# In[203]:


performance_week_category = performance_week_category.reset_index()


# In[204]:


performance_week_category['previous_week_sales'] = performance_week_category.groupby(['category_name'])['weekly_sales'].transform('shift')
performance_week_category['previous_week_transaction_ct'] = performance_week_category.groupby(['category_name'])['weekly_transaction_ct'].transform('shift')
performance_week_category['weekly_sales_increase_rate'] = performance_week_category['weekly_sales'] /performance_week_category['previous_week_sales'] - 1
performance_week_category['weekly_transaction_ct_increase_rate'] = performance_week_category['weekly_transaction_ct'] /performance_week_category['previous_week_transaction_ct'] - 1


# In[205]:


performance_week_category.tail(15)


# In[206]:


performance_week_category.to_csv('performance_week_category.csv')


# In[207]:


performance_week_category_pivot = performance_week_category[performance_week_category['First_day_of_the_week'] != '2016-08-22']                                    .pivot_table(index = ['First_day_of_the_week'], columns = ['category_name'], values = 'weekly_transaction_ct', aggfunc = sum)


# In[208]:


performance_week_category_pivot = performance_week_category_pivot.reset_index()


# In[209]:


performance_week_category_pivot.columns = ['First_day_of_the_week', 'Games', 'Photos & Videos', 'Utilities','Entertainment', 'Social Networking']


# In[210]:


performance_week_category_pivot['First_day_of_the_week']


# In[211]:


f, ax = plt.subplots(1, 1)
ax.plot_date(performance_week_category_pivot['First_day_of_the_week'], performance_week_category_pivot['Games'], color="blue", label="Weekly Number of Games Transactions", linestyle="-")
ax.plot_date(performance_week_category_pivot['First_day_of_the_week'], performance_week_category_pivot['Photos & Videos'], color="orange", label="Weekly Number of Photos & Videos Transactions", linestyle="-")
ax.plot_date(performance_week_category_pivot['First_day_of_the_week'], performance_week_category_pivot['Utilities'], color="purple", label="Weekly Number of Utilities Transactions", linestyle="-")
ax.plot_date(performance_week_category_pivot['First_day_of_the_week'], performance_week_category_pivot['Entertainment'], color="yellow", label="Weekly Number of Entertainment Transactions", linestyle="-")
ax.plot_date(performance_week_category_pivot['First_day_of_the_week'], performance_week_category_pivot['Social Networking'], color="brown", label="Weekly Number of Entertainment Transactions", linestyle="-")

# ax.legend()
ax.set_title('Weekly Transaction Trends', fontsize=20)
ax.set_xlabel("Week", fontsize = 16)
ax.set_ylabel("Number of Transactions", fontsize = 16)

plt.gcf().autofmt_xdate()
plt.show()


# In[ ]:





# #### Weekly sales and transactions in each category are quite stable over the analysis period (two weeks have missing days).

# In[212]:


ax = sns.barplot(x = 'category_name', y = 'weekly_sales_increase_rate', data = performance_week_category[performance_week_category['First_day_of_the_week'] == '2016-09-12'])
ax.set_title('Week over Week Sales Trends', fontsize=20)
ax.set_xlabel("Category", fontsize = 16)
ax.set_ylabel("Weekly Sales Amount Change Rate", fontsize = 16)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set(ylim=(-0.05, 0.05))


# In[213]:


ax = sns.barplot(x = 'category_name', y = 'weekly_transaction_ct_increase_rate', data = performance_week_category[performance_week_category['First_day_of_the_week'] == '2016-09-12'])
ax.set_title('Week over Week Transaction Trends', fontsize=20)
ax.set_xlabel("Category", fontsize = 16)
ax.set_ylabel("Weekly Transaction Change Rate", fontsize = 16)
ax.set(ylim=(-0.05, 0.05))


# ## Performance of day of week by category

# In[214]:


performance_weekday = transaction_df_final.groupby(['First_day_of_the_week','transaction_weekday_nm']).agg(daily_sales = ('price', sum), active_acct_ct = ('acct_id', 'nunique'))
performance_weekday['daily_transaction_ct'] = transaction_df_final.groupby(['First_day_of_the_week','transaction_weekday_nm']).size()


# In[215]:


performance_weekday = performance_weekday.reset_index()


# In[216]:


performance_weekday.head()


# In[217]:


performance_weekday['previous_weekday_active_acct_ct'] = performance_weekday.groupby(['transaction_weekday_nm'])['active_acct_ct'].transform('shift')
performance_weekday['previous_weekday_sales'] = performance_weekday.groupby(['transaction_weekday_nm'])['daily_sales'].transform('shift')
performance_weekday['previous_weekday_transaction_ct'] = performance_weekday.groupby(['transaction_weekday_nm'])['daily_transaction_ct'].transform('shift')
performance_weekday['wow_active_acct_ct'] = performance_weekday['active_acct_ct']/performance_weekday['previous_weekday_active_acct_ct'] - 1
performance_weekday['wow_sales_increase_rate'] = performance_weekday['daily_sales'] /performance_weekday['previous_weekday_sales'] - 1
performance_weekday['wow_transaction_ct_increase_rate'] = performance_weekday['daily_transaction_ct'] /performance_weekday['previous_weekday_transaction_ct'] - 1


# In[218]:


performance_weekday.tail(20)


# In[219]:


performance_weekday.to_csv('performance_weekday.csv')


# #### Weekly sales and transactions in each category are quite stable over the analysis period (two weeks have missing days).

# In[220]:


ax = sns.barplot(x = 'transaction_weekday_nm', y = 'wow_sales_increase_rate', data = performance_weekday[performance_weekday['First_day_of_the_week'] == '2016-09-12'])
ax.set_title('Week over Week Sales Trends by Weekday', fontsize=20)
ax.set_xlabel("Weekday", fontsize = 16)
ax.set_ylabel("Weekly Sales Amount Change Rate", fontsize = 16)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set(ylim=(-0.05, 0.05))


# In[221]:


ax = sns.barplot(x = 'transaction_weekday_nm', y = 'wow_transaction_ct_increase_rate', data = performance_weekday[performance_weekday['First_day_of_the_week'] == '2016-09-12'])
ax.set_title('Week over Week Sales Trends by Weekday', fontsize=20)
ax.set_xlabel("Weekday", fontsize = 16)
ax.set_ylabel("Weekly Transaction Change Rate", fontsize = 16)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set(ylim=(-0.05, 0.05))


# In[222]:


ax = sns.barplot(x = 'transaction_weekday_nm', y = 'wow_active_acct_ct', data = performance_weekday[performance_weekday['First_day_of_the_week'] == '2016-09-12'])
ax.set_title('Week over Week Sales Trends by Weekday', fontsize=20)
ax.set_xlabel("Weekday", fontsize = 16)
ax.set_ylabel("Weekly Sales Amount Change Rate", fontsize = 16)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set(ylim=(-0.05, 0.05))


# ### Popular Apps of each week

# In[223]:


app_week= transaction_df_final.groupby(['First_day_of_the_week', 'app_name','category_name']).agg(app_weekly_sales = ('price', sum), app_active_acct_ct = ('acct_id', 'nunique'))
app_week['app_weekly_transaction_ct'] = transaction_df_final.groupby(['First_day_of_the_week', 'app_name','category_name']).size()
app_week['app_weekly_sales_rank'] = app_week.groupby(['First_day_of_the_week'])['app_weekly_sales'].rank(method = 'min', ascending = False)
app_week['app_weekly_transaction_ct_rank'] = app_week.groupby(['First_day_of_the_week'])['app_weekly_transaction_ct'].rank(method = 'min', ascending = False)
app_week['app_weekly_sales_rank_category'] = app_week.groupby(['First_day_of_the_week','category_name'])['app_weekly_sales'].rank(method = 'min', ascending = False)
app_week['app_weekly_transaction_ct_rank_category'] = app_week.groupby(['First_day_of_the_week','category_name'])['app_weekly_transaction_ct'].rank(method = 'min', ascending = False)
app_week = app_week.sort_values(by = ['First_day_of_the_week','app_weekly_sales_rank','app_weekly_transaction_ct_rank']).reset_index()


# In[224]:


app_week['category_weekly_sales'] = app_week.groupby(['First_day_of_the_week', 'category_name'])['app_weekly_sales'].transform(sum)
app_week['category_weekly_sales_rank'] = app_week.groupby(['First_day_of_the_week'])['category_weekly_sales'].rank(method = 'min', ascending = False)


# In[225]:


app_week['app_weekly_sales_rank'] = app_week['app_weekly_sales_rank'].astype('int')
app_week['app_weekly_transaction_ct_rank'] = app_week['app_weekly_transaction_ct_rank'].astype('int')
app_week['app_weekly_sales_rank_category'] = app_week['app_weekly_sales_rank_category'].astype('int')
app_week['app_weekly_transaction_ct_rank_category'] = app_week['app_weekly_transaction_ct_rank_category'].astype('int')


# In[226]:


app_week[((app_week['app_weekly_sales_rank'] <= 5) | (app_week['app_weekly_transaction_ct_rank'] <= 5)) & (app_week['First_day_of_the_week'] == app_week['First_day_of_the_week'].max())]


# In[227]:


app_week.head()


# In[228]:


app_week.to_csv('app_week.csv')


# In[229]:


category_rank = app_week[((app_week['app_weekly_sales_rank_category'] <= 5) | (app_week['app_weekly_transaction_ct_rank_category'] <= 5)) & (app_week['First_day_of_the_week'] == app_week['First_day_of_the_week'].max())]


# In[230]:


category_rank.head()


# In[231]:


category_rank.sort_values(by = ['category_weekly_sales_rank', 'app_weekly_sales_rank_category'])[category_rank['category_name'] == 'Games'][['category_name','app_name']]


# In[232]:


category_rank.sort_values(by = ['category_weekly_sales_rank', 'app_weekly_sales_rank_category'])[category_rank['category_name'] == 'Photos & Videos'][['category_name','app_name']]


# In[233]:


category_rank.sort_values(by = ['category_weekly_sales_rank', 'app_weekly_sales_rank_category'])[['category_name','app_name']]


# ## 2.4 Monthly Trends

# In[234]:


performance= transaction_df_final.groupby(['transaction_year_month']).agg(monthly_sales = ('price', sum),                                                                     active_acct_ct = ('acct_id', 'nunique'),                                                                     purchased_unique_app = ('parent_app_content_id', 'nunique'),                                                                   active_days = ('transaction_dt','nunique'))


# In[235]:


performance['number of transaction'] = transaction_df_final.groupby(['transaction_year_month']).size()
performance['avg daily number of transaction'] = performance['number of transaction']/ performance['active_days']
performance['avg daily sales amount'] = performance['monthly_sales']/ performance['active_days']


# In[236]:


transaction_df_final.groupby(['transaction_year_month']).agg('size')


# In[237]:


performance = performance.reset_index()


# In[238]:


performance


# In[239]:


ax = sns.barplot(x=performance['transaction_year_month'], y= 'monthly_sales', data=performance.sort_values(by=['transaction_year_month']))
ax.set_title('Monthly Sales Trends', fontsize=20)
ax.set_xlabel("Month", fontsize = 16)
ax.set_ylabel("Monthly Sales Amount", fontsize = 16)
ax.yaxis.set_major_formatter('$ {x}')
for p in ax.patches:
    ax.annotate("$ " + format(round(p.get_height()/1000000,2), '.2f')+"M", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# In[240]:


ax = sns.barplot(x=performance['transaction_year_month'], y= 'avg daily sales amount', data=performance.sort_values(by=['transaction_year_month']))
ax.set_title('Avg Daily Sales Trends by Month', fontsize=20)
ax.set_xlabel("Month", fontsize = 16)
ax.set_ylabel("Avg Daily Sales Amount", fontsize = 16)
ax.yaxis.set_major_formatter('$ {x}')
for p in ax.patches:
    ax.annotate("$ "+ format(round(p.get_height()/1000,2), '.2f')+"K", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# ### Hypothesis: The daily sales of July 2016 is higher than other months, which was driven by the great sales performance on July 4th. Is that true? 

# In[241]:


performance2= transaction_df_final[transaction_df_final['transaction_dt'] !='2016-07-04'].groupby(['transaction_year_month']).agg(monthly_sales = ('price', sum),                                                                     active_acct_ct = ('acct_id', 'nunique'),                                                                     purchased_unique_app = ('parent_app_content_id', 'nunique'),                                                                   active_days = ('transaction_dt','nunique'))


# In[242]:


performance2['number of transaction'] = transaction_df_final[transaction_df_final['transaction_dt'] !='2016-07-04'].groupby(['transaction_year_month']).size()
performance2['avg daily number of transaction'] = performance2['number of transaction']/ performance2['active_days']
performance2['avg daily sales amount'] = performance2['monthly_sales']/ performance2['active_days']


# In[243]:


performance2 = performance2.reset_index()


# In[244]:


performance2


# In[245]:


ax = sns.barplot(x=performance2['transaction_year_month'], y= 'avg daily sales amount', data=performance2)
ax.set_title('Avg Daily Sales Trends by Month', fontsize=20)
ax.set_xlabel("Month", fontsize = 16)
ax.set_ylabel("Adjusted Avg Daily Sales Amount", fontsize = 16)
ax.yaxis.set_major_formatter('$ {x}')
for p in ax.patches:
    ax.annotate("$ "+ format(round(p.get_height()/1000,2), '.2f')+"K", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# #### The daily sales of July 2016 is higher than other months, even removing the impact from the great sales performance on July 4th. 

# # 3 Insights

# ## 3.1 Analyze App category

# In[246]:


ax = sns.countplot(x='category_name', data=transaction_df_final)
ax.set_title('Transaction Count by Category', fontsize=20)
ax.set_xlabel("Category", fontsize = 16)
ax.set_ylabel("Transaction Count", fontsize = 16)
for p in ax.patches:
    ax.annotate(format(round(p.get_height()/1000,2), '.1f')+"K", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# In[247]:


category_summary = transaction_df_final.groupby(['category_name']).agg(total_revenue= ('price',sum))
category_summary['total_transaction_ct'] = transaction_df_final.groupby(['category_name']).size()
category_summary['avg_transaction_amt'] = category_summary['total_revenue'] / category_summary['total_transaction_ct']
category_summary['avg_transaction_amt_index'] = category_summary['avg_transaction_amt']/category_summary['avg_transaction_amt'].min()
category_summary['total_transaction_ct'] = category_summary['total_transaction_ct'].astype('float')


# In[248]:


category_summary


# In[249]:


category_summary.to_csv('category_summary.csv')


# In[250]:


fig1, ax1 = plt.subplots()
ax1.pie(category_summary.total_revenue, explode= (0, 0.1, 0, 0, 0) , labels=category_summary.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Total Revenue by Category', fontsize=20)

plt.show()


# In[251]:


fig1, ax1 = plt.subplots()
ax1.pie(category_summary.total_transaction_ct, explode= (0, 0.1, 0, 0, 0) , labels=category_summary.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Total Transaction Count by Category', fontsize=20)

plt.show()


# In[252]:


ax = sns.barplot(x= category_summary.index, y = 'avg_transaction_amt', data = category_summary)
ax.set_title('Average Transaction Amount by Category', fontsize=20)
ax.yaxis.set_major_formatter('$ {x}')
ax.set_xlabel("Category", fontsize = 16)
ax.set_ylabel("Avg Transaction Amount", fontsize = 16)
for p in ax.patches:
    ax.annotate("$ " + format(round(p.get_height(),2), '.2f'), (p.get_x()+0.4, p.get_height()*0.98), ha='center', va='top', color='white', size=12)


# #### - Games category is the most important App category with more than half of the share in transaction count and revenue. 
# #### - The most important is that games category has the highest average transaction amount (\\$ 11.40). It almost doubled the average transaction amount in utilities category (\\$ 5.81).

# ## 3.2 Analyze transaction device

# In[253]:


transaction_df_final.head()


# In[254]:


ax = sns.countplot(x='transaction_device_name', data=transaction_df_final)
ax.set_title('Transaction Count by Device', fontsize=20)
ax.set_xlabel("Device", fontsize = 16)
ax.set_ylabel("Transaction Count", fontsize = 16)
# for p in ax.patches:
#     ax.annotate(format(round(p.get_height()/1000), '.0f')+"K", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# In[255]:


device_summary = transaction_df_final.groupby(['transaction_device_name']).agg(total_revenue= ('price',sum))
device_summary['total_transaction_ct'] = transaction_df_final.groupby(['transaction_device_name']).size()
device_summary['avg_transaction_amt'] = device_summary['total_revenue'] / device_summary['total_transaction_ct']
device_summary['transaction_share'] = device_summary['total_transaction_ct'] / device_summary['total_transaction_ct'].sum()
device_summary['revenue_share'] = device_summary['total_revenue'] / device_summary['total_revenue'].sum()
device_summary['avg_transaction_amt_index'] = device_summary['avg_transaction_amt']/device_summary['avg_transaction_amt'].min()
device_summary['total_transaction_ct'] = device_summary['total_transaction_ct'].astype('float')


# In[256]:


device_summary


# In[257]:


device_summary.to_csv('device_summary.csv')


# In[258]:


ax = sns.barplot(x = device_summary.index, y = 'revenue_share', data = device_summary)
ax.set_title('App Sales by Device', fontsize=20)
ax.set_xlabel("Device", fontsize = 16)
ax.set_ylabel("Sales Share", fontsize = 16)
for p in ax.patches:
    ax.annotate(format(round(p.get_height(),2)*100, '.0f')+"%", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# In[259]:


ax = sns.barplot(x = device_summary.index, y = 'transaction_share', data = device_summary)
ax.set_title('Transaction by Device', fontsize=20)
ax.set_xlabel("Device", fontsize = 16)
ax.set_ylabel("Transaction Share", fontsize = 16)
for p in ax.patches:
    ax.annotate(format(round(p.get_height(),2)*100, '.0f')+"%", (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# #### 72% of Apps are purchased with iphone, which generated 73% of the revenue.

# ### Check number of Apps by device

# In[260]:


app_df.head()


# In[261]:


ax = sns.countplot(x = 'app_device_name', data = app_df)
ax.set_title('App Count by Device', fontsize=20)
ax.set_xlabel("Device", fontsize = 16)
ax.set_ylabel("Transaction Count", fontsize = 16)
for p in ax.patches:
    ax.annotate(format(round(p.get_height(),2), '.0f'), (p.get_x()+0.4, p.get_height()*0.9), ha='center', va='top', color='white', size=12)


# In[262]:


app_device = app_df.groupby(['app_device_name']).agg(app_ct = ('parent_app_content_id', 'nunique'))


# In[263]:


app_device


# In[264]:


app_device.to_csv('app_device.csv')


# In[265]:


fig1, ax1 = plt.subplots()
ax1.pie(app_device.app_ct, explode= (0.1, 0, 0) , labels=app_device.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Total Transaction Count by Category', fontsize=20)

plt.show()


# #### 82% of Apps are avalible on both devices

# ### Check the device preference for apps can be used on both iphone and ipad

# In[266]:


device_summary2 = transaction_df_final[transaction_df_final['app_device_name'] == 'Both'].groupby(['transaction_device_name']).agg(total_revenue= ('price',sum))
device_summary2['total_transaction_ct'] = transaction_df_final.groupby(['transaction_device_name']).size()
device_summary2['avg_transaction_amt'] = device_summary2['total_revenue'] / device_summary2['total_transaction_ct']
device_summary2['transaction_share'] = device_summary2['total_transaction_ct'] / device_summary2['total_transaction_ct'].sum()
device_summary2['revenue_share'] = device_summary2['total_revenue'] / device_summary2['total_revenue'].sum()
device_summary2['avg_transaction_amt_index'] = device_summary2['avg_transaction_amt']/device_summary2['avg_transaction_amt'].min()
device_summary2['total_transaction_ct'] = device_summary2['total_transaction_ct'].astype('float')


# In[267]:


device_summary2


# In[268]:


device_summary2.to_csv('device_summary2.csv')


# In[269]:


fig1, ax1 = plt.subplots()
ax1.pie(device_summary2.total_revenue, explode= (0, 0.1) , labels=device_summary2.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Total Revenue by Device', fontsize=20)

plt.show()


# In[270]:


fig1, ax1 = plt.subplots()
ax1.pie(device_summary2.total_transaction_ct, explode= (0, 0.1) , labels=device_summary2.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Total Transaction Count by Device', fontsize=20)

plt.show()


# In[271]:


ax = sns.barplot(x= device_summary2.index, y = 'avg_transaction_amt', data = device_summary2)
ax.set_title('Average Transaction Amount by Device', fontsize=20)
ax.yaxis.set_major_formatter('$ {x}')
ax.set_xlabel("Category", fontsize = 16)
ax.set_ylabel("Avg Transaction Amount", fontsize = 16)
for p in ax.patches:
    ax.annotate("$ " + format(round(p.get_height(),2), '.2f'), (p.get_x()+0.4, p.get_height()*0.98), ha='center', va='top', color='white', size=12)


# #### - Among Apps that could be used on both iphone and ipad, people are more likely to purchase them with iPhone. 
# #### - Among Apps that could be used on both iphone and ipad, Apps purchased via iPhone is relatively more expensive compared to those purchased with iPad. 
# #### - That means the marketing campaign and purchase process on iPhone is more important. 

# ## 3.3 Payment Type

# ### Hypothesis testing : will payment method on file reduce the possibility to be a dormant account? 

# In[272]:


account_summary.head()


# In[273]:


txn_acct_payment = account_summary[(~account_summary['acct_id'].isin(outliers1)) & (~account_summary['acct_id'].isin(outliers2))].groupby(['payment_type','paid_acct_flag']).agg(no_txn_acct_ct = ('acct_id', 'nunique')).reset_index()


# In[274]:


txn_acct_payment.head()


# In[275]:


fig1, ax1 = plt.subplots()
ax1.pie(txn_acct_payment[txn_acct_payment['payment_type']=='Free only'].no_txn_acct_ct, explode= (0,0.1) , labels= txn_acct_payment[txn_acct_payment['payment_type']=='Free only'].paid_acct_flag, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['blue','grey'], textprops={'color':"w",'size':16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Active Account Proportion of Free Only Accounts', fontsize=24)
plt.legend(loc="best")
plt.show()


# In[276]:


fig1, ax1 = plt.subplots()
ax1.pie(txn_acct_payment[txn_acct_payment['payment_type']=='PMOF'].no_txn_acct_ct, explode= (0, 0.1) , labels= txn_acct_payment[txn_acct_payment['payment_type']=='PMOF'].paid_acct_flag, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = ['blue','grey'], textprops={'color':"w",'size':16})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Active Account Proportion of PMOF Accounts', fontsize=24)
plt.legend(loc="best")
plt.show()


# ### Hypothesis testing : will payment method on file increase the possibility to download/purchase an App? 

# In[277]:


ax = sns.countplot(x= 'payment_type', data = transaction_df_final)
ax.set_title('Number of Transaction by Account Pyament Method', fontsize=20)
ax.yaxis.set_major_formatter('$ {x}')
ax.set_xlabel("Account Pyament Method", fontsize = 16)
ax.set_ylabel("Number of Transaction", fontsize = 16)
for p in ax.patches:
    ax.annotate(format(round(p.get_height()), '.0f'), (p.get_x()+0.4, p.get_height()*0.98), ha='center', va='top', color='white', size=12)


# In[278]:


account_summary.head()


# In[279]:


account_summary['avg_spend_amount'] = account_summary['total_revenue']/account_summary['total_txn']
account_summary['acct_tenure'] = transaction_df_final['transaction_dt'].max() - account_summary['account_create_dt']
account_summary['acct_tenure_days'] = account_summary['acct_tenure']/np.timedelta64(1,'D')
account_summary['acct_tenure_months'] = account_summary['acct_tenure']/np.timedelta64(1,'M')
account_summary['acct_tenure_years'] = account_summary['acct_tenure']/np.timedelta64(1,'Y')


# In[280]:


tenure_cut_labels = ['transactions without account','1 month', '2- 6 months', '6 months - 1 year', '1- 3 years','3 - 5 years', '5 years+']
tenure_cut_bins = [transaction_df_final['acct_tenure_months'].min(), 0, 1, 6, 12,36, 60, transaction_df_final['acct_tenure_months'].max()]
account_summary['Tenure_Band'] = pd.cut(account_summary['acct_tenure_months'], bins = tenure_cut_bins, labels = tenure_cut_labels)


# In[281]:


account_summary.head()


# In[282]:


payment1 = account_summary.groupby(['payment_type']).agg(total_revenue = ('total_revenue',sum),total_txn = ('total_txn',sum), total_acct_ct = ('acct_id','count'))
payment1['avg_txn'] = payment1['total_txn']/payment1['total_acct_ct']


# In[283]:


payment1


# In[284]:


payment1.to_csv('payment1.csv')


# In[285]:


payment2 = account_summary.groupby(['payment_type','paid_acct_flag']).agg(total_revenue = ('total_revenue',sum),total_txn = ('total_txn',sum), total_acct_ct = ('acct_id','count'))
payment2['avg_txn'] = payment2['total_txn']/payment2['total_acct_ct']


# In[286]:


payment2


# In[287]:


payment2.to_csv('payment2.csv')


# #### - Payment method on file will not affect the possibility to download an App

# ### Hypothesis testing : will payment method on file increase the possibility to download more Apps? 

# H0 : PMOF will not affect the number of apps downloaded. 
# The average number of apps downloaded by free only accounts is the same as average number of apps downloaded by PMOF accounts.
# 
# H1 : PMOF will affect the number of apps downloaded. 
# The average number of apps downloaded by free only accounts is not the same as average number of apps downloaded by PMOF accounts.

# Control Group: free only accounts
# Test Group: PMOF accounts

# In[288]:


control = account_summary[account_summary['payment_type'] == 'Free only']['total_txn']
test = account_summary[account_summary['payment_type'] == 'PMOF']['total_txn']


# In[289]:


mean_control = control.mean()
mean_test = test.mean()


# In[290]:


n_control = len(control)
n_test = len(test)


# n_control = 50000 > 30, large sample

# n_test = 50000 > 30, large sample

# Since population variance is unknown and the sample size is big, this would be a t-test.

# Assumed alpha is 5% and beta is 80%

# In[291]:


alpha = 0.05
beta = 0.8


# In[292]:


d_hat = mean_test - mean_control


# In[293]:


def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = sum(deviations) / n
    return variance


# In[294]:


variance_control = variance(control)
variance_test = variance(test)


# In[295]:


print("variance of the control and test group is: ",round(np.sqrt(variance_control),4),round(np.sqrt(variance_test),4))


# In[296]:


n_control = len(control)


# In[297]:


variance_pool = ((n_control - 1) * variance_control + (n_test - 1) * variance_test) / (n_control - 1 + n_test -1)


# In[298]:


print(variance_pool)


# In[299]:


SE = np.sqrt(variance_pool) * np.sqrt((1/n_control + 1/n_test))


# In[300]:


t_threshold = 1.96 
# 95% confidence interval


# In[301]:


margin_of_error = SE * t_threshold


# In[302]:


lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error


# In[303]:


print(f'Confidence interval is {round(d_hat - margin_of_error,4)} to {round(d_hat + margin_of_error,4)}')


# #### These two groups are extremely statistically significant.
# #### In other words, adding PMOF on account are more likely to download more Apps. Motivating users to add payment method could increase account activity. 

# ## 3.4 InApp Content Type

# ### Hypothesis testing : will subscribtion apps increase revenue? 

# In[304]:


ax = sns.countplot(x= 'inapp_content_type', data = transaction_df_final)
ax.set_title('Number of Transaction by Content Type', fontsize=20)
ax.set_xlabel("Content Type", fontsize = 16)
ax.set_ylabel("Number of Transaction", fontsize = 16)


# In[305]:


app_summary = app_df.groupby(['category_name','inapp_content_type']).agg(app_ct= ('parent_app_content_id','nunique'),content_ct = ('content_id','nunique')).reset_index()


# In[306]:


app_summary['category_app_ct'] = app_summary.groupby(['category_name'])['app_ct'].transform('sum')
app_summary['total_app_ct'] = app_summary['app_ct'].sum()
app_summary['category_app_ct%'] = app_summary['app_ct']/app_summary['category_app_ct']
app_summary['total_app_ct%'] = app_summary['app_ct']/app_summary['total_app_ct']


# In[307]:


app_summary


# In[308]:


app_summary.to_csv('app_summary.csv')


# #### The duplicates come from the unknown app_name. We need to use column 'parent_app_content_id' to identify each unique App.

# In[309]:


ax = sns.barplot(x= 'inapp_content_type', y = 'app_ct', data = app_summary, hue = 'category_name')
ax.set_title('Number of Apps by Content Content Type', fontsize=20)
ax.set_xlabel("Number of Apps", fontsize = 16)
ax.set_ylabel("Content Type", fontsize = 16)


# #### Since games has no subscription Apps and the number of Apps in social entworking is too small, I will focus on the other three categores. 

# ### Subscription transaction vs consumable transaction

# In[310]:


subscription = transaction_df_final[(transaction_df_final['category_name'] != 'Games') & (transaction_df_final['category_name'] != 'Social Networking')]                [['app_name', 'content_id','category_name','parent_app_content_id','inapp_content_type','acct_id','price','transaction_year_month']]


# In[311]:


subscription.head()


# In[312]:


subscription_summary = subscription.groupby(['inapp_content_type']).agg(app_name_ct = ('app_name', 'nunique'),                                                           parent_app_content_ct = ('parent_app_content_id', 'nunique'),                                                          content_ct = ('content_id', 'nunique'),                                                          sales_amount = ('price',sum))


# In[313]:


subscription_summary['transaction_ct'] = subscription.groupby(['inapp_content_type']).size()
subscription_summary['avg_paid_content_ct_per_app'] = subscription_summary['content_ct']/subscription_summary['parent_app_content_ct']
subscription_summary['avg_revenue_per_app'] = subscription_summary['sales_amount']/subscription_summary['parent_app_content_ct']
subscription_summary['avg_price_per_txn'] = subscription_summary['sales_amount']/subscription_summary['transaction_ct']
subscription_summary['avg_txn_per_app'] = subscription_summary['transaction_ct']/subscription_summary['parent_app_content_ct']
subscription_summary['avg_revenue_per_content'] = subscription_summary['sales_amount']/subscription_summary['content_ct']
subscription_summary['avg_txn_per_content'] = subscription_summary['transaction_ct']/subscription_summary['content_ct']


# Assumed consumable App is a one-time purchase while subscription would be a repeated purchase. 

# For the same group of accounts, they would only pay for each consumable App once and will continue to pay for sbscription App in a year. 

# In that case, I will add a time ratio to current subscription sales. 

# In[314]:


retention_rate = 0.6
# Assumed 60% of subscribers will finish the subscription


# In[315]:


subscription_summary['estimated_annual_sales_amount'] = np.where(subscription_summary.index == 'subscription', subscription_summary['sales_amount'] * (1/ratio) * retention_rate, subscription_summary['sales_amount'])
subscription_summary['estimated_annual_transaction_ct'] = np.where(subscription_summary.index == 'subscription', subscription_summary['transaction_ct'] * (1/ratio) * retention_rate, subscription_summary['transaction_ct'])

subscription_summary['estimated_annual_revenue_per_app'] = np.where(subscription_summary.index == 'subscription', subscription_summary['avg_revenue_per_app'] * (1/ratio)  *retention_rate, subscription_summary['avg_revenue_per_app'])
subscription_summary['estimated_annual_txn_per_app'] = np.where(subscription_summary.index == 'subscription', subscription_summary['avg_txn_per_app'] * (1/ratio) * retention_rate, subscription_summary['avg_txn_per_app'])

subscription_summary['estimated_annual_revenue_per_content'] = np.where(subscription_summary.index == 'subscription', subscription_summary['avg_revenue_per_content'] * (1/ratio) * retention_rate, subscription_summary['avg_revenue_per_content'])
subscription_summary['estimated_annual_txn_per_content'] = np.where(subscription_summary.index == 'subscription', subscription_summary['avg_txn_per_content'] * (1/ratio)  *retention_rate, subscription_summary['avg_txn_per_content'])


# In[316]:


subscription_summary


# In[317]:


subscription_summary.to_csv('subscription_summary.csv')


# In[318]:


ax = sns.barplot(x= subscription_summary.index, y = 'estimated_annual_revenue_per_app', data = subscription_summary)
ax.set_title('Estimated Annual Revenue per App by Content Type', fontsize=20)
ax.set_xlabel("Content Type", fontsize = 12)
ax.set_ylabel("Estimated Annual Revenue Amount", fontsize = 12)
ax.yaxis.set_major_formatter(ticker.EngFormatter())


# In[319]:


ax = sns.barplot(x= subscription_summary.index, y = 'estimated_annual_revenue_per_content', data = subscription_summary)
ax.set_title('Estimated Annual Revenue per content by Content Type', fontsize=20)
ax.set_xlabel("Content Type", fontsize = 12)
ax.set_ylabel("Estimated Annual Revenue Amount", fontsize = 12)
ax.yaxis.set_major_formatter(ticker.EngFormatter())


# In[320]:


ax = sns.barplot(x= subscription_summary.index, y = 'estimated_annual_txn_per_app', data = subscription_summary)
ax.set_title('Estimated Annual Transaction per App by Content Type', fontsize=20)
ax.set_xlabel("Content Type", fontsize = 12)
ax.set_ylabel("Estimated Annual Transaction", fontsize = 12)
ax.yaxis.set_major_formatter(ticker.EngFormatter())


# In[321]:


ax = sns.barplot(x= subscription_summary.index, y = 'estimated_annual_txn_per_content', data = subscription_summary)
ax.set_title('Estimated Annual Transaction per content by Content Type', fontsize=20)
ax.set_xlabel("Content Type", fontsize = 12)
ax.set_ylabel("Estimated Annual Transaction", fontsize = 12)
ax.yaxis.set_major_formatter(ticker.EngFormatter())


# In[322]:


subscription_content = subscription.groupby(['content_id','inapp_content_type']).agg(total_revenue = ('price',sum))
subscription_content['transaction_ct'] = subscription.groupby(['content_id','inapp_content_type']).size()


# In[323]:


subscription_content = subscription_content.reset_index()


# In[324]:


subscription_content.sort_values(by='transaction_ct').head()


# ### Hypothesis testing : will subscription increase the average number of transaction of content? 

# H0 : Content type will not affect the average number of transaction of content. 
# The average number of transactions of consumable content is the same as average number of transactions of subscription content.
# 
# H1 : Content type will affect the average number of transaction of content. 
# The average number of transactions of consumable content is not the same as average number of transactions of subscription content.

# Control Group: consumable App content;
# Test Group: subscription App content

# In[325]:


control = subscription_content[subscription_content['inapp_content_type'] == 'consumable']['transaction_ct']
test = subscription_content[subscription_content['inapp_content_type'] == 'subscription']['transaction_ct']


# In[326]:


mean_control = control.mean()
mean_test = test.mean()


# In[327]:


n_control = len(control)
n_test = len(test)


# In[328]:


print("sample size in control and test groups is: ", n_control, "and", n_test)


# n_control = 979 > 30, large sample; n_test = 257 > 30, large sample

# since population variance is unknown and the sample size is big, this would be a t-test.

# Assumed alpha is 5% and beta is 80%

# In[329]:


alpha = 0.05
beta = 0.8


# In[330]:


d_hat = mean_test - mean_control


# In[331]:


variance_control = variance(control)
variance_test = variance(test)


# In[332]:


print("variance of the control and test group is: ",round(np.sqrt(variance_control),4),round(np.sqrt(variance_test),4))


# In[333]:


variance_pool = ((n_control - 1) * variance_control + (n_test - 1) * variance_test) / (n_control - 1 + n_test -1)


# In[334]:


SE = np.sqrt(variance_pool) * np.sqrt((1/n_control + 1/n_test))


# In[335]:


t_threshold = 1.96 
# 95% confidence interval


# In[336]:


margin_of_error = SE * t_threshold


# In[337]:


lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error


# In[338]:


print(f'Confidence interval is {round(d_hat - margin_of_error,4)} to {round(d_hat + margin_of_error,4)}')


# #### These two groups are not statistically significant.
# #### In other words, subscription will  not increase the number of transaction of content. 
# #### There is alo possibility there is a sampling bais since we only have limited data (less than 4 months and less than 1000 Apps). 

# ### Hypothesis testing : will subscription increase the average revenue of content? 

# H0 : Content type will not affect the average revenue of content. 
# The average revenue of consumable content is the same as average revenue of subscription content.
# 
# H1 : Content type will affect the average revenue of content. 
# The average revenue of consumable content is not the same as average revenue of subscription content.

# Control Group: consumable App content;
# Test Group: subscription App content

# In[339]:


control = subscription_content[subscription_content['inapp_content_type'] == 'consumable']['total_revenue']
test = subscription_content[subscription_content['inapp_content_type'] == 'subscription']['total_revenue']


# In[340]:


mean_control = control.mean()
mean_test = test.mean()


# In[341]:


n_control = len(control)
n_test = len(test)


# In[342]:


print("sample size in control and test groups is: ", n_control, "and", n_test)


# n_control = 979 > 30, large sample; n_test = 257 > 30, large sample

# since population variance is unknown and the sample size is big, this would be a t-test.

# Assumed alpha is 5% and beta is 80%

# In[343]:


alpha = 0.05
beta = 0.8


# In[344]:


d_hat = mean_test - mean_control


# In[345]:


variance_control = variance(control)
variance_test = variance(test)


# In[346]:


print("variance of the control and test group is: ",round(np.sqrt(variance_control),4),round(np.sqrt(variance_test),4))


# In[347]:


variance_pool = ((n_control - 1) * variance_control + (n_test - 1) * variance_test) / (n_control - 1 + n_test -1)


# In[348]:


SE = np.sqrt(variance_pool) * np.sqrt((1/n_control + 1/n_test))


# In[349]:


t_threshold = 1.96 
# 95% confidence interval


# In[350]:


margin_of_error = SE * t_threshold


# In[351]:


lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error


# In[352]:


print(f'Confidence interval is {round(d_hat - margin_of_error,4)} to {round(d_hat + margin_of_error,4)}')


# ### Subscription by category

# In[353]:


subscription_category = subscription.groupby(['category_name','inapp_content_type']).agg(app_name_ct = ('app_name', 'nunique'),                                                           parent_app_content_ct = ('parent_app_content_id', 'nunique'),                                                          content_ct = ('content_id', 'nunique'),                                                          sales_amount = ('price',sum))


# In[354]:


subscription_category['transaction_ct'] = subscription.groupby(['category_name','inapp_content_type']).size()
subscription_category['avg_paid_content_ct_per_app'] = subscription_category['content_ct']/subscription_category['parent_app_content_ct']
subscription_category['avg_revenue_per_app'] = subscription_category['sales_amount']/subscription_category['parent_app_content_ct']
subscription_category['avg_price_per_txn'] = subscription_category['sales_amount']/subscription_category['transaction_ct']
subscription_category['avg_txn_per_app'] = subscription_category['transaction_ct']/subscription_category['parent_app_content_ct']
subscription_category['avg_revenue_per_content'] = subscription_category['sales_amount']/subscription_category['content_ct']
subscription_category['avg_txn_per_content'] = subscription_category['transaction_ct']/subscription_category['content_ct']


# In[355]:


subscription_category = subscription_category.reset_index()


# In[356]:


subscription_category.head()


# Assumed consumable App is a one-time purchase while subscription would be a repeated purchase. 

# For the same group of accounts, they would only pay for each consumable App once and will continue to pay for sbscription App in a year. 

# In that case, I will add a time ratio to current subscription sales. 

# In[357]:


subscription_category['estimated_annual_sales_amount'] = np.where(subscription_category['inapp_content_type'] == 'subscription', subscription_category['sales_amount'] * (1/ratio) * retention_rate, subscription_category['sales_amount'])
subscription_category['estimated_annual_transaction_ct'] = np.where(subscription_category['inapp_content_type'] == 'subscription', subscription_category['transaction_ct'] * (1/ratio) * retention_rate, subscription_category['transaction_ct'])

subscription_category['estimated_annual_revenue_per_app'] = np.where(subscription_category['inapp_content_type'] == 'subscription', subscription_category['avg_revenue_per_app'] * (1/ratio) * retention_rate, subscription_category['avg_revenue_per_app'])
subscription_category['estimated_annual_txn_per_app'] = np.where(subscription_category['inapp_content_type'] == 'subscription', subscription_category['avg_txn_per_app'] * (1/ratio) * retention_rate, subscription_category['avg_txn_per_app'])

subscription_category['estimated_annual_revenue_per_content'] = np.where(subscription_category['inapp_content_type'] == 'subscription', subscription_category['avg_revenue_per_content'] * (1/ratio) * retention_rate, subscription_category['avg_revenue_per_content'])
subscription_category['estimated_annual_txn_per_content'] = np.where(subscription_category['inapp_content_type'] == 'subscription', subscription_category['avg_txn_per_content'] * (1/ratio) * retention_rate, subscription_category['avg_txn_per_content'])


# In[358]:


subscription_category


# In[359]:


subscription_category.to_csv('subscription_category.csv')


# #### Subscription in entertainment, Photos & Videos and utilities category can be more profitable compared to cosnumable. 
# #### All the games apps are consumable. Providing games apps with subscription options may be a good idea. 

# In[360]:


subscription_pivot = subscription.pivot_table(index = ['acct_id','content_id'], columns = ['transaction_year_month'], values = ['price'], aggfunc = sum).fillna(0)
subscription_pivot = subscription_pivot.loc[~(subscription_pivot==0).all(axis=1)]


# In[361]:


subscription_pivot.columns = subscription_pivot.columns.droplevel(0)


# In[362]:


subscription_pivot = subscription_pivot.reset_index()


# In[363]:


subscription_pivot.head()


# In[364]:


subscription_pivot.columns = ['acct_id', 'content_id','2016-06', '2016-07', '2016-08', '2016-09']


# In[365]:


subscription_pivot['Jun_purchase'] = np.where(subscription_pivot['2016-06']>0 , 1,0)
subscription_pivot['Jul_purchase'] = np.where(subscription_pivot['2016-07']>0 , 1,0)
subscription_pivot['Aug_purchase'] = np.where(subscription_pivot['2016-08']>0 , 1,0)
subscription_pivot['Sep_purchase'] = np.where(subscription_pivot['2016-09']>0 , 1,0)
subscription_pivot['purchase_month_ct'] = subscription_pivot['Jun_purchase'] + subscription_pivot['Jul_purchase'] + subscription_pivot['Aug_purchase']+ subscription_pivot['Sep_purchase']
subscription_pivot['repeat_purchase'] = np.where((subscription_pivot['purchase_month_ct'])>1 , 1,0)


# In[366]:


first_txn_dt = transaction_df_final.groupby(['acct_id','content_id']).agg(first_txn_dt = ('transaction_dt',min))


# In[367]:


first_txn_dt = first_txn_dt.reset_index()


# In[368]:


subscription_pivot = subscription_pivot.merge(first_txn_dt, on =['acct_id','content_id'], how = 'left')                                       .merge(app_df, on ='content_id', how = 'left')


# In[369]:


subscription_pivot[subscription_pivot['repeat_purchase']== 1].head()


# In[370]:


subscription_pivot_summary = subscription_pivot.groupby(['category_name','inapp_content_type']).agg(acct_content_ct = ('acct_id','count'), content_ct = ('content_id','nunique'), repeat_purchase_acct_ct = ('repeat_purchase', 'sum'))


# In[371]:


subscription_pivot_summary['avg_purchase_acct_per_content'] = subscription_pivot_summary['acct_content_ct']/subscription_pivot_summary['content_ct']
subscription_pivot_summary['avg_repeat_purchase_acct_per_content'] = subscription_pivot_summary['repeat_purchase_acct_ct']/subscription_pivot_summary['content_ct']


# In[372]:


subscription_pivot_summary


# In[373]:


subscription_pivot_summary.to_csv('subscription_pivot_summary.csv')


# In[374]:


subscription_ab_test_entertainment = subscription_pivot[(subscription_pivot['category_name'] == 'Entertainment') ]                                                            .groupby(['content_id','inapp_content_type']).agg(repeat_purchase_acct_ct = ('repeat_purchase', sum)).reset_index()
subscription_ab_test_utilities = subscription_pivot[(subscription_pivot['category_name'] == 'Utilities')]                                                            .groupby(['content_id','inapp_content_type']).agg(repeat_purchase_acct_ct = ('repeat_purchase', sum)).reset_index()
subscription_ab_test_PhotosVideos = subscription_pivot[(subscription_pivot['category_name'] == 'Photos & Videos') ]                                                            .groupby(['content_id','inapp_content_type']).agg(repeat_purchase_acct_ct = ('repeat_purchase', sum)).reset_index()


# In[375]:


subscription_ab_test_entertainment.head()


# In[376]:


subscription_ab_test_utilities.head()


# ### Hypothesis testing : will subscription increase the possibility to repeat purchase in entertainment content? 

# H0 : Content type will not affect if an account will repeat purchase a content or not. 
# The average number of repeat purchased account of consumable Apps is the same as average number ofrepeat purchased account of subscription Apps.
# 
# H1 : Content type will affect if an account will repeat purchase a content or not. 
# The average number of repeat purchased account of consumable Apps is not the same as average number ofrepeat purchased account of subscription Apps.

# Control Group: consumable App content;
# Test Group: subscription App content

# In[377]:


control = subscription_ab_test_entertainment[subscription_ab_test_entertainment['inapp_content_type'] == 'consumable']['repeat_purchase_acct_ct']
test = subscription_ab_test_entertainment[subscription_ab_test_entertainment['inapp_content_type'] == 'subscription']['repeat_purchase_acct_ct']


# In[378]:


mean_control = control.mean()
mean_test = test.mean()


# In[379]:


n_control = len(control)
n_test = len(test)


# In[380]:


print("sample size in control and test groups is: ", n_control, "and", n_test)


# n_control = 184 > 30, large sample; n_test = 75 > 30, large sample

# since population variance is unknown and the sample size is big, this would be a t-test.

# Assumed alpha is 5% and beta is 80%

# In[381]:


alpha = 0.05
beta = 0.8


# In[382]:


d_hat = mean_test - mean_control


# In[383]:


variance_control = variance(control)
variance_test = variance(test)


# In[384]:


print("variance of the control and test group is: ",round(np.sqrt(variance_control),4),round(np.sqrt(variance_test),4))


# In[385]:


variance_pool = ((n_control - 1) * variance_control + (n_test - 1) * variance_test) / (n_control - 1 + n_test -1)


# In[386]:


SE = np.sqrt(variance_pool) * np.sqrt((1/n_control + 1/n_test))


# In[387]:


t_threshold = 1.96 
# 95% confidence interval


# In[388]:


margin_of_error = SE * t_threshold


# In[389]:


lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error


# In[390]:


print(f'Confidence interval is {round(d_hat - margin_of_error,4)} to {round(d_hat + margin_of_error,4)}')


# #### These two groups are not statistically significant.
# #### In other words, subscription will not increase the repeat purchase in entertainment Apps. 
# #### There is alo possibility there is a sampling bais since we only have limited data (less than 4 months and less than 1000 Apps). 

# ## Hypothesis testing : will subscription increase the possibility to repeat purchase in utilities content? 

# H0 : Content type will not affect if an account will repeat purchase a content or not. 
# The average number of repeat purchased account of consumable Apps is the same as average number ofrepeat purchased account of subscription Apps.
# 
# H1 : Content type will affect if an account will repeat purchase a content or not. 
# The average number of repeat purchased account of consumable Apps is not the same as average number ofrepeat purchased account of subscription Apps.

# Control Group: consumable App content;
# Test Group: subscription App content

# In[391]:


control = subscription_ab_test_utilities[subscription_ab_test_utilities['inapp_content_type'] == 'consumable']['repeat_purchase_acct_ct']
test = subscription_ab_test_utilities[subscription_ab_test_utilities['inapp_content_type'] == 'subscription']['repeat_purchase_acct_ct']


# In[392]:


mean_control = control.mean()
mean_test = test.mean()


# In[393]:


n_control = len(control)
n_test = len(test)


# In[394]:


print("sample size in control and test groups is: ", n_control, "and", n_test)


# n_control = 210 > 30, large sample; n_test = 61 > 30, large sample

# since population variance is unknown and the sample size is big, this would be a t-test.

# Assumed alpha is 5% and beta is 80%

# In[395]:


alpha = 0.05
beta = 0.8


# In[396]:


d_hat = mean_test - mean_control


# In[397]:


variance_control = variance(control)
variance_test = variance(test)


# In[398]:


print("variance of the control and test group is: ",round(np.sqrt(variance_control),4),round(np.sqrt(variance_test),4))


# In[399]:


variance_pool = ((n_control - 1) * variance_control + (n_test - 1) * variance_test) / (n_control - 1 + n_test -1)


# In[400]:


SE = np.sqrt(variance_pool) * np.sqrt((1/n_control + 1/n_test))


# In[401]:


t_threshold = 1.96 
# 95% confidence interval


# In[402]:


margin_of_error = SE * t_threshold


# In[403]:


lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error


# In[404]:


print(f'Confidence interval is {round(d_hat - margin_of_error,4)} to {round(d_hat + margin_of_error,4)}')


# #### These two groups are statistically significant.
# #### In other words, subscription will increase the repeat purchase in utilities Apps. 

# ### Hypothesis testing : will subscription increase the possibility to repeat purchase in Photos & Videos content? 

# H0 : Content type will not affect if an account will repeat purchase a content or not. 
# The average number of repeat purchased account of consumable Apps is the same as average number ofrepeat purchased account of subscription Apps.
# 
# H1 : Content type will affect if an account will repeat purchase a content or not. 
# The average number of repeat purchased account of consumable Apps is not the same as average number ofrepeat purchased account of subscription Apps.

# Control Group: consumable App content;
# Test Group: subscription App content

# In[405]:


control = subscription_ab_test_PhotosVideos[subscription_ab_test_PhotosVideos['inapp_content_type'] == 'consumable']['repeat_purchase_acct_ct']
test = subscription_ab_test_PhotosVideos[subscription_ab_test_PhotosVideos['inapp_content_type'] == 'subscription']['repeat_purchase_acct_ct']


# In[406]:


mean_control = control.mean()
mean_test = test.mean()


# In[407]:


n_control = len(control)
n_test = len(test)


# In[408]:


print("sample size in control and test groups is: ", n_control, "and", n_test)


# n_control = 210 > 30, large sample; n_test = 61 > 30, large sample

# Since population variance is unknown and the sample size is big, this would be a t-test.

# Assumed alpha is 5% and beta is 80%

# In[409]:


alpha = 0.05
beta = 0.8


# In[410]:


d_hat = mean_test - mean_control


# In[411]:


variance_control = variance(control)
variance_test = variance(test)


# In[412]:


print("variance of the control and test group is: ",round(np.sqrt(variance_control),4),round(np.sqrt(variance_test),4))


# In[413]:


variance_pool = ((n_control - 1) * variance_control + (n_test - 1) * variance_test) / (n_control - 1 + n_test -1)


# In[414]:


SE = np.sqrt(variance_pool) * np.sqrt((1/n_control + 1/n_test))


# In[415]:


t_threshold = 1.96 
# 95% confidence interval


# In[416]:


margin_of_error = SE * t_threshold


# In[417]:


lower_bound = d_hat - margin_of_error
upper_bound = d_hat + margin_of_error


# In[418]:


print(f'Confidence interval is {round(d_hat - margin_of_error,4)} to {round(d_hat + margin_of_error,4)}')


# #### These two groups are not statistically significant.
# #### In other words, subscription will not increase the repeat purchase in photos & videos Apps. 
# #### There is alo possibility there is a sampling bais since we only have limited data (less than 4 months and less than 1000 Apps). 

# ## Repeat purchase possibility in category

# In[419]:


transaction_df_final[['app_name', 'content_id','category_name','parent_app_content_id','inapp_content_type','acct_id','price','transaction_year_month']].head()


# In[420]:


repeat_purchase_pivot = transaction_df_final.pivot_table(index = ['acct_id','content_id','category_name'], columns = ['transaction_year_month'], values = ['price'], aggfunc = sum).fillna(0)


# In[421]:


repeat_purchase_pivot.head()


# In[422]:


repeat_purchase_pivot  = repeat_purchase_pivot .loc[~(repeat_purchase_pivot ==0).all(axis=1)]
repeat_purchase_pivot.columns = repeat_purchase_pivot.columns.droplevel(0)
repeat_purchase_pivot = repeat_purchase_pivot.reset_index()


# In[423]:


repeat_purchase_pivot.head()


# In[424]:


repeat_purchase_pivot.columns = ['acct_id', 'content_id','category_name','2016-06', '2016-07', '2016-08', '2016-09']


# In[425]:


repeat_purchase_pivot['Jun_purchase'] = np.where(repeat_purchase_pivot['2016-06']>0 , 1,0)
repeat_purchase_pivot['Jul_purchase'] = np.where(repeat_purchase_pivot['2016-07']>0 , 1,0)
repeat_purchase_pivot['Aug_purchase'] = np.where(repeat_purchase_pivot['2016-08']>0 , 1,0)
repeat_purchase_pivot['Sep_purchase'] = np.where(repeat_purchase_pivot['2016-09']>0 , 1,0)
repeat_purchase_pivot['purchase_month_ct'] = repeat_purchase_pivot['Jun_purchase'] + repeat_purchase_pivot['Jul_purchase'] + repeat_purchase_pivot['Aug_purchase']+ repeat_purchase_pivot['Sep_purchase']
repeat_purchase_pivot['repeat_purchase'] = np.where((repeat_purchase_pivot['purchase_month_ct'])>1 , 1,0)


# In[426]:


repeat_purchase_pivot[repeat_purchase_pivot['repeat_purchase']== 1].head()


# In[427]:


repeat_purchase_summary = repeat_purchase_pivot.groupby(['category_name']).agg(repeat_purchase_ct = ('repeat_purchase','count'), content_ct = ('content_id','nunique'))


# In[428]:


repeat_purchase_summary['avg_repeat_purchase_ct'] = repeat_purchase_summary['repeat_purchase_ct']/repeat_purchase_summary['content_ct']


# In[429]:


repeat_purchase_summary


# ### Games content demostrated the highest repeat purchase times

# In[430]:


repeat_purchase_summary.to_csv('repeat_purchase_summary.csv')


# ## 4. Other Analysis

# ## 4.1 The relationship between price change (promotion)  and sales/transaction

# In[431]:


regular_price = transaction_df_final.groupby(['content_id']).agg(regular_price = ('price',statistics.mode))


# In[432]:


price = transaction_df_final.groupby(['content_id','price']).agg(price_occur_days = ('transaction_dt','nunique'))


# In[433]:


price['most_days'] = price.groupby(['content_id','price'])['price_occur_days'].max()


# In[434]:


regular_price = price[price['price_occur_days'] == price['most_days']].reset_index()


# In[435]:


regular_price.rename(columns = {'price':'regular price'}, inplace = True)
regular_price.drop(['price_occur_days','most_days'],axis = 1, inplace = True)


# In[436]:


regular_price.head()


# In[437]:


transaction_df_final = transaction_df_final.merge(regular_price, on = 'content_id', how = 'left')


# In[438]:


transaction_df_final['promotion_flag'] = np.where(transaction_df_final['price'] < transaction_df_final['regular price'],1,0)
transaction_df_final['promotion_rate'] = 1- transaction_df_final['price']/transaction_df_final['regular price']


# In[439]:


transaction_df_final.head()


# In[440]:


transaction_df_final[transaction_df_final['promotion_flag'] == 1]


# #### There is no promotion data

# ## 4.2 The relationship between account tenure and spend amount/number of transaction

# In[441]:


transaction_tenure = transaction_df_final.groupby(['Tenure_Band']).agg(acct_ct = ('acct_id','nunique'), revenue = ('price',sum), free_transaction_ct = ('free_transaction_flag', sum))
transaction_tenure['number of transaction'] = transaction_df_final.groupby(['Tenure_Band']).size()


# In[442]:


transaction_tenure['avg free_transaction_ct'] =  transaction_tenure['free_transaction_ct'] / transaction_tenure['acct_ct']
transaction_tenure['avg number of transaction'] = transaction_tenure['number of transaction'] / transaction_tenure['acct_ct']
transaction_tenure['avg revenue amount'] = transaction_tenure['revenue'] / transaction_tenure['acct_ct']
transaction_tenure['avg transaction amount'] = transaction_tenure['avg revenue amount']/ transaction_tenure['avg number of transaction']
transaction_tenure['paid transaction rate'] = 1 - transaction_tenure['avg free_transaction_ct']/ transaction_tenure['avg number of transaction']


# In[443]:


transaction_tenure['transaction share'] = transaction_tenure['number of transaction']/transaction_tenure['number of transaction'].sum()
transaction_tenure['free transaction share'] = transaction_tenure['free_transaction_ct']/transaction_tenure['free_transaction_ct'].sum()
transaction_tenure['revenue_share'] = transaction_tenure['revenue']/transaction_tenure['revenue'].sum()


# In[444]:


transaction_tenure


# #### Accounts with long tenure are more likely to download more Apps.  Accounts with short tenure are more likely to download fewer Apps and most them are free Apps.
# #### Accounts with 5 years tenure are more likely to purchase paid Apps. It is likely to have selection bais. 

# In[445]:


transaction_tenure_price = transaction_df_final.groupby(['Tenure_Band','Price_Band']).agg(acct_ct = ('acct_id','nunique'), revenue = ('price',sum))
transaction_tenure_price['number of transaction'] = transaction_df_final.groupby(['Tenure_Band','Price_Band']).size()
transaction_tenure_price['avg transaction amount']  = transaction_tenure_price['revenue'] /transaction_tenure_price['number of transaction'] 


# In[446]:


transaction_tenure_price


# #### Only accounts with more than 3 years tenure have purchased App during the analysis period.

# # 4.3 The relationship between number of device and sales/transaction

# In[447]:


transaction_df_final.head()


# In[448]:


device_ct = transaction_df_final.groupby(['acct_id']).agg(device_ct = ('transaction_device_id', 'nunique'))


# In[449]:


device_ct['device_ct'].unique()


# #### All the accounts, who have transaction during the 110 days analysis period, all have transactions on both iPhone and iPad.

# ## 4.4 New App/content activities

# In[450]:


app_first_dt = transaction_df_final.groupby(['parent_app_content_id','app_name','content_id']).agg(first_txn_dt = ('transaction_dt', min))


# In[451]:


app_first_dt['first_txn_dt'].unique()


# #### All the App/content are available on or before June 1st 2016. There was no new App/content start to be available during the 110 analysis period. 

# # Thank you!
