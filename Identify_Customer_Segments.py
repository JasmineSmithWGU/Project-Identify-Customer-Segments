#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA    
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of

#891221 persons (rows) and 85 features (columns)
azdias.shape


# In[4]:


# rows and columns, print the first few rows).
azdias.head(10)


# In[5]:


#85 features (rows) x 4 columns

feat_info.shape


# In[6]:


#names of the 85 features
azdias.head()


# In[7]:


#The 4 columns contain information about the features
feat_info.head(n=10)


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[8]:


# Identify missing or unknown data values and convert them to NaNs.

azdias.isnull().sum()


# In[9]:


feat_info['missing_or_unknown'] = feat_info['missing_or_unknown'].apply(lambda x: x[1:-1].split(sep=','))


# In[10]:


#replace null values with Nan

azdias_missing_vals = azdias.copy()
for attribute, missing_vals in zip(feat_info['attribute'], feat_info['missing_or_unknown']):
    if missing_vals[0] != '':
            azdias_missing_vals[attribute] = azdias_missing_vals[attribute].replace(missing_vals, np.nan)
            for value in missing_vals:
                if value.isnumeric() or value.lstrip('-').isnumeric():
                    value = int(value)
                azdias_missing_vals.loc[azdias_missing_vals[attribute] == value, attribute] = np.nan


# In[11]:


#count number of overall missing values that are over 0.
azdias.isnull().sum().sum()


# In[12]:


#Azdias dataset with converted missing or unknown values to Nan 

azdias = azdias_missing_vals
azdias.head()


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[13]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

#columns in azdias dataset that have missing data
azdias.isnull().sum()


# In[14]:


#all columns with missing or unknown data. Seperated the nan values from the dataset to identify only the columns with missing data only.
azdias_null = azdias.isnull().sum().sort_values(ascending=True)
azdias_null = azdias_null[azdias_null > 0]
print(azdias_null)


# In[15]:


# Investigate patterns in the amount of missing data in each column.
# There is a few outliers that is skewing the data right that are above 200000. 

azdias.isnull().sum().hist()
plt.title('Azdias Missing data')
plt.ylabel('Column Count')
plt.xlabel('Count of missing data')


# In[16]:


#drop outlier columns and sort columns with missing or unknown data to show better visual.

azdias_null = azdias_null.drop(azdias_null[azdias_null>200000].index)
azdias_null.sort_values(ascending=True)
#Dropped values over 200000 that are outliers and resorted dataset to show values without outliers.


# In[17]:


azdias_null.hist()
plt.title('Azdias Missing data')
plt.ylabel('Column Count')
plt.xlabel('Count of missing data')

#Dataset is now skewed to the left showing that most of the columns with missing data are in the 100,000 range.


# In[18]:


azdias_null.plot.barh(figsize=(20,30))
plt.xlabel('Count of missing values for each column')
plt.ylabel('column name with missing values')
plt.show()


# In[19]:


# How much data is missing in each row of the dataset?
azdias_null.sum()

#There is a sum of 5035304 missing or unknown values after dropping the outliers. There was originally 8,373,929 before dropping the outliers. The outliers were a total of 338625.


# In[20]:


#Data missing in each row
azdias_null


# In[21]:


#Drop outliers from original dataset
azdias_nan=azdias.isnull().sum()
columns = azdias_nan[azdias_nan>200000].index
azdias.drop(columns=columns, axis="columns", inplace=True)


# In[22]:


azdias.head()


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# (Double click this cell and replace this text with your own text, reporting your observations regarding the amount of missing data in each column. Are there any patterns in missing values? Which columns were removed from the dataset?)
# 
# After evaluating the data, there were 8,373,929 values that were null. When reviewing the histogram most columns had less than 200000 values. There were 6 columns that were outliers that effected the histogram. By removing those 6 columns, we able to pull a report over columns that have under 200000 null values (not including the nonnull columns). After dropping the outliers, the histogram skewed from the right to the left and showed that the majority of columns with missing values have a range within 100,000. After dropping the outliers that had 3,338,625 missing values there were a total of 5,035,304 missing or unknown values that was able to be used within a dataset that no longer has outliers. 

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[23]:


azdias_null_rows = azdias.isnull().sum(axis=1)
azdias_null_rows.describe()


# In[24]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
less_null = azdias[azdias.isnull().sum(axis=1) <= 10]
more_null = azdias[azdias.isnull().sum(axis=1) > 10]


# In[25]:


columns=azdias_null[azdias_null<200000].index


# In[26]:


columns


# In[27]:


for col in columns [:5]:
    fig, axes = plt.subplots(1,2, figsize=(14, 6), sharey=True)
    sns.countplot(less_null[col], ax=axes[0], color='g').set(title= 'Less Null Values')
    sns.countplot(more_null[col], ax=axes[1], color='r').set(title= 'More Null Values')


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# Based on the rows with less missing values vs more missing values we can see that there is a significant diffence between the two that are presented. Most rows have less than 10 null values.

# In[28]:


less_null.shape


# In[29]:


more_null.shape


# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[30]:


# How many features are there of each data type?

feat_info.type.value_counts()


# In[31]:


#Remove non-numerical features
feat_info=feat_info[feat_info.attribute.isin(less_null.columns)]


# In[32]:


feat_info.type.value_counts()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[33]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
cat_var = feat_info[feat_info.type == "categorical"].attribute
print(cat_var)


# In[34]:



binary_cat=[]
multi_cat=[]

for col in cat_var:
    if less_null[col].nunique()==2:
        binary_cat.append(col)
    else:
        multi_cat.append(col)


# In[35]:


for col in binary_cat:
    print(less_null[col].value_counts())


# In[36]:


for col in multi_cat:
    print(less_null[col].value_counts())


# In[37]:


# Re-encode categorical variable(s) to be kept in the analysis.
less_null["ANREDE_KZ"].replace({2:0,1:1},inplace=True)
less_null["SOHO_KZ"].replace({0.0:0,1.0:1},inplace=True)
less_null["VERS_TYP"].replace({2:0,1:1},inplace=True)
less_null["OST_WEST_KZ"].replace({'W':0,'O':1},inplace=True)
less_null["OST_WEST_KZ"].astype('int',inplace=True)
less_null=pd.get_dummies(data=less_null,columns=multi_cat)


# In[38]:


#for feature in multi_cat:
 #   less_null=less_null.drop(feature, axis=1)


# In[39]:


for col in binary_cat:
    print(less_null[col].value_counts())


# In[40]:


less_null.info()


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# Binary categorical features have been standardized and one-hot-encoding has been used on the multi-level categorical features.
# 

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[41]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
less_null.PRAEGENDE_JUGENDJAHRE.head()


# In[42]:


less_null['DECADE'] = less_null['PRAEGENDE_JUGENDJAHRE']
less_null['MOVEMENT'] = less_null['PRAEGENDE_JUGENDJAHRE']


# In[43]:


DECADE = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5, 13:5, 14:6, 15:6}
MOVEMENT = {1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:0, 8:1, 9:0, 10:1, 11:0, 12:1, 13:0, 14:1, 15:0}


# In[44]:


less_null['DECADE'].replace(DECADE, inplace=True)
less_null['MOVEMENT'].replace(MOVEMENT, inplace=True)


# In[45]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
less_null['WEALTH'] = less_null['CAMEO_INTL_2015']
less_null['LIFE_STAGE'] = less_null['CAMEO_INTL_2015']


# In[46]:


WEALTH = {'11':1, '12':1, '13':1, '14':1, '15':1, '21':2, '22':2, '23':2, '24':2, '25':2,
               '31':3, '32':3, '33':3, '34':3, '35':3, '41':4, '42':4, '43':4, '44':4, '45':4,
               '51':5, '52':5, '53':5, '54':5, '55':5}

LIFE_STAGE = {'11':1, '12':2, '13':3, '14':4, '15':5, '21':1, '22':2, '23':3, '24':4, '25':5,
                   '31':1, '32':2, '33':3, '34':4, '35':5, '41':1, '42':2, '43':3, '44':4, '45':5,
                   '51':1, '52':2, '53':3, '54':4, '55':5}


# In[47]:


less_null['WEALTH'].replace(WEALTH, inplace=True)
less_null['LIFE_STAGE'].replace(LIFE_STAGE, inplace=True)


# In[48]:


less_null


# In[49]:


#less_null = less_null.drop(['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015', 'PLZ8_BAUMAX', 'LP_LEBENSPHASE_FEIN', 'WOHNLAGE', 'REGIOTYP','KKK'], axis=1)


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# Created two new feature columns, and copied values from the initial mixed-value feature column. Performed a replace operation using the dictionaries on the new feature columns. Removed the multi-level categorical features.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[50]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

mixed_val_feat = feat_info[feat_info.type == 'mixed'].attribute
print(mixed_val_feat)


# In[51]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

for feature in mixed_val_feat:
    less_null.drop(feature, axis=1, inplace=True)


# In[52]:


less_null.head()


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[53]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    #replace null values with Nan

    azdias_missing_vals = azdias.copy()
    for attribute, missing_vals in zip(feat_info['attribute'], feat_info['missing_or_unknown']):
        if missing_vals[0] != '':
                azdias_missing_vals[attribute] = azdias_missing_vals[attribute].replace(missing_vals, np.nan)
                for value in missing_vals:
                    if value.isnumeric() or value.lstrip('-').isnumeric():
                        value = int(value)
                    azdias_missing_vals.loc[azdias_missing_vals[attribute] == value, attribute] = np.nan
    
    #drop outlier columns and sort columns with missing or unknown data to show better visual.
    azdias_null = azdias.isnull().sum().sort_values(ascending=True)
    azdias_null = azdias_null[azdias_null > 0]
    azdias_null = azdias_null.drop(azdias_null[azdias_null>200000].index)
    

    # Re-encode categorical variable(s) to be kept in the analysis.
    less_null = azdias[azdias.isnull().sum(axis=1) <= 10]


       # drop multi-leve features
            
    cat_var = feat_info[feat_info.type == "categorical"].attribute
    
    binary_cat=[]
    multi_cat=[]

    for col in cat_var:
        if less_null[col].nunique()==2:
            binary_cat.append(col)
        else:
            multi_cat.append(col)    
            
        #for feature in multi_cat:
         #   less_null=less_null.drop(feature, axis=1)
    
    less_null["ANREDE_KZ"].replace({2:0,1:1},inplace=True)
    less_null["SOHO_KZ"].replace({0.0:0,1.0:1},inplace=True)
    less_null["VERS_TYP"].replace({2:0,1:1},inplace=True)
    less_null["OST_WEST_KZ"].replace({'W':0,'O':1},inplace=True)
    less_null["OST_WEST_KZ"].astype('int',inplace=True)
    less_null=pd.get_dummies(data=less_null,columns=multi_cat)
    
    # Investigate "PRAGENDE_JUGENDJAHRE" and engineer two new variables.
    #less_null = azdias[azdias.isnull().sum(axis=1) <= 10]
    columns=azdias_null[azdias_null<200000].index
    less_null['DECADE'] = less_null['PRAEGENDE_JUGENDJAHRE']
    less_null['MOVEMENT'] = less_null['PRAEGENDE_JUGENDJAHRE']
    
    #Create dict
    DECADE = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5, 13:5, 14:6, 15:6}
    MOVEMENT = {1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:0, 8:1, 9:0, 10:1, 11:0, 12:1, 13:0, 14:1, 15:0}
    
    #Replace with new variables
    less_null['DECADE'].replace(DECADE, inplace=True)
    less_null['MOVEMENT'].replace(MOVEMENT, inplace=True)
      
    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    less_null['WEALTH'] = less_null['CAMEO_INTL_2015']
    less_null['LIFE_STAGE'] = less_null['CAMEO_INTL_2015']
                                  
    WEALTH = {'11':1, '12':1, '13':1, '14':1, '15':1, '21':2, '22':2, '23':2, '24':2, '25':2,
               '31':3, '32':3, '33':3, '34':3, '35':3, '41':4, '42':4, '43':4, '44':4, '45':4,
               '51':5, '52':5, '53':5, '54':5, '55':5}

    LIFE_STAGE = {'11':1, '12':2, '13':3, '14':4, '15':5, '21':1, '22':2, '23':3, '24':4, '25':5,
                   '31':1, '32':2, '33':3, '34':4, '35':5, '41':1, '42':2, '43':3, '44':4, '45':5,
                   '51':1, '52':2, '53':3, '54':4, '55':5}
    #Drop from original data set
    #less_null = less_null.drop(['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015', 'PLZ8_BAUMAX', 'LP_LEBENSPHASE_FEIN', 'WOHNLAGE', 'REGIOTYP','KKK'], axis=1)
    
    #Replace with new variables
    less_null['WEALTH'].replace(WEALTH, inplace=True)
    less_null['LIFE_STAGE'].replace(LIFE_STAGE, inplace=True)
    
    #Drop mixed features 
    mixed_val_feat = feat_info[feat_info["type"]=="mixed"]["attribute"]
    for feature in mixed_val_feat:
        less_null.drop(feature, axis=1, inplace=True)                             
                                  
    # Return the cleaned dataframe.
    return less_null
    


# In[54]:


less_null.info()


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[55]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.

less_null.isnull().sum().sum()


# In[56]:


fill_missing = Imputer(strategy='most_frequent')
azdias_clean_imputed = pd.DataFrame(fill_missing.fit_transform(less_null))


# In[57]:


azdias_clean_imputed.isnull().sum().sum() 


# In[58]:


azdias_clean_imputed.head()


# In[59]:


azdias_clean_imputed.columns = less_null.columns
azdias_clean_imputed.index = less_null.index


# In[60]:


# Apply feature scaling to the general population demographics data.
scaler = StandardScaler()
azdias_scaled = scaler.fit_transform(azdias_clean_imputed)


# In[61]:


azdias_scaled = pd.DataFrame(azdias_scaled, columns=list(azdias_clean_imputed))


# In[62]:


azdias_scaled.head()


# ### Discussion 2.1: Apply Feature Scaling
# 
# By using the imputer method I was able to replace all missing values with the most frequent value. After, I scaled all the features using the StandardScaler() and renaming the dataframe to azdias_scaled.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[63]:


# Apply PCA to the data.
pca = PCA()
pca.fit(azdias_scaled)
pca.explained_variance_ratio_


# In[64]:


# Investigate the variance accounted for by each principal component.

plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.title("Variance by component")
plt.xlabel("Principal component")
plt.ylabel("Variance Ratio")
plt.show()


# In[65]:


# Re-apply PCA to the data while selecting for number of components to retain.

pca_80 = PCA(n_components=80)
azdias_PCA = pca_80.fit_transform(azdias_scaled)


# In[66]:


for i in np.arange(10, 81, 10):
    print('{} components explain {} of variance.'.format(i, pca.explained_variance_ratio_[:i].sum()))


# In[67]:


pca_80.components_.shape


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# Based on the bar chart, I decided to retain 80 principal components since we will be able to still capture atleast 78%.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[68]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

def weights(pca, i):
    df = pd.DataFrame(pca.components_, columns=list(azdias_scaled.columns))
    weights = df.iloc[i].sort_values(ascending=False)
    return weights

weights0 = weights(pca_80, 0)
weights0


# In[69]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

weights1 = weights(pca_80, 1)
weights1


# In[70]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

weights2 = weights(pca_80, 2)
weights2


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[71]:


# Over a number of different cluster counts...


    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.
    
azdias_PCA = azdias_PCA[np.random.choice(azdias_PCA.shape[0], int(azdias_PCA.shape[0]*0.25), replace=False)]
azdias_PCA 

sse = [] # Sum of Squared Errors
k_range = list(range(1, 20))

for k in k_range:
    kmeans = KMeans(k, random_state=1234, max_iter=30, n_jobs=-1).fit(azdias_PCA)
    score = np.abs(kmeans.score(azdias_PCA))
    sse.append(score)
    #print('Clustering done for {} k, with SSE {}'.format(k, score))


# In[72]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

plt.xticks(np.arange(0, k_range[-1]+1, step=1))
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('SSE vs. K')
plt.plot(k_range, sse, linestyle='-', marker='o');


# In[73]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
k = 24
kmeans = KMeans(k, random_state=1234, max_iter=30, n_jobs=-1).fit(azdias_PCA)
population_clusters = kmeans.predict(azdias_PCA)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[74]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', delimiter=';')
customers.head(10)


# In[75]:


customers.info()


# In[76]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.


customers_clean = clean_data(customers)
    
    


# In[77]:



Customers_imputed = pd.DataFrame(fill_missing.fit_transform(customers_clean))
Customers_imputed.columns = customers_clean.columns
Customers_imputed.index = customers_clean.index


# In[78]:


Customers_imputed.head()


# In[79]:


customers_clean_std = scaler.transform(Customers_imputed)


# In[80]:


customers_pca = pca.transform(Customers_imputed)


# In[81]:


kmeans = KMeans(n_clusters = 14).fit(customers_pca)
customer_clusters = kmeans.predict(customers_pca)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[90]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

figure, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
figure.subplots_adjust(hspace = 1, wspace=.3)

sns.countplot(customer_clusters, ax=axs[0])
axs[0].set_title('Customer Clusters')
sns.countplot(population_clusters, ax=axs[1])
axs[1].set_title('Population Clusters')


# In[177]:


# What kinds of people are part of a cluster that is overrepresented in the
overrepresented = scaler.inverse_transform(pca.inverse_transform(customers_pca[np.where(customer_clusters==14)])).round()

overrep = pd.DataFrame(data=overrep, columns=customers_clean.columns)
overrep


# In[173]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?

underrepresented = scaler.inverse_transform(pca.inverse_transform(customers_pca[np.where(customer_clusters==14)])).round()
underrep = pd.DataFrame(data=underrep, columns=customers_clean.columns)

underrep.head()


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




