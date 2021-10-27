# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:35:37 2021

@author: Michael Impey
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

#===============================================================
#Functions
#===============================================================

def pause():
    '''used to pause program to view output'''
    input('===> Press Return to Continue Program ?') 

def draw_missing_data_table(df):
    '''Create table for missing data analysis'''
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return round(missing_data,2)

def import_file(filename):
    '''Import data - import and set up data frames'''
    file = pd.read_csv(filename)
    print("\nTrain.info() in imported file:\n", file.info())
    print("\nID.info() in imported file:\n", file.info()) 
    return file
 
def DF_data_types(df):
    '''#find dataframe d.types'''
    # select numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    print("\nNumeric Column labels: \n", numeric_cols)
    # select non numeric columns
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values
    print("\nNon-Numeric Column labels: \n", non_numeric_cols)
    # display number of columns to see if any missing
    print("\nNumber of Columns: " + str(len(df.columns)) + \
          "\nNumber of Numeric columns: " + str(len(numeric_cols)) + \
              "\nNumber of Non-Numeric columns: " + str(len(non_numeric_cols)))
    #Confirm number of Columns correct
    try:
        assert (len(numeric_cols) + len(non_numeric_cols)) == (len(df.columns))
        print("\nNumber of columns = MATCH\n")
    except:
        print("\nNumber of columns = DO NOT MATCH\n")
        pause()
    return numeric_cols, non_numeric_cols 

def EDA_Visual(df):
    ''' Exploratory Data - visualise'''
    #Missing data
    msno.matrix(df)
    plt.show()
    # Plot the correlations as a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2g')
    plt.show()
    
def EDA_Descriptive(df):
    '''Exploratory Data - descriptive'''
    print("\nDataFrame shape():\n",df.shape)
    print("\nDataFrame info():\n",df.info())
    #cannot see all data so need to print per column
    for col in df.columns:
        print("\nColumn ["+str(col)+"] describe(all):\n", df[col].describe(include = 'all'))

def dataframe_unique_check(df):
    '''unique and Unique count'''
    n=20# to set the lenght of the series to see what in in column
    for col in df.columns:
        unique = df[col].unique()
        print("\n\n\n>> Column "+str(col)+" Unique contents for first "+str(n)+\
              ":\n" + str(unique[:n]))
        print("\n>> Column "+str(col)+" Unique contents & count:\n",df[col].value_counts())
        
def plots_box(col):
    # matplotlib Boxplot
    df.boxplot(column=[col])
    plt.title("\nBox Plot for Column: "+str(col))
    plt.show()

def plots_hist(col):    
    # matplotlib histogram
    plt.hist(df[col], color = 'blue', edgecolor = 'black')
    plt.title("\nHistorgam Plot for Column: "+str(col))
    plt.show()
    
def most_important_features(X,y):
    #Feature Importance - https://www.kaggle.com/niklasdonges/end-to-end-project-with-python
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X, y)
    importances = pd.DataFrame({'feature':X.columns,'importance':np.round(random_forest.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print('\nimportances.head(15):\n',importances.head(15))    
    #show bar plot on impotant features
    importances.plot.bar()
    plt.show()
        
def  transform_categorical_variables(dataframe):
    ''' Transform categorical variables into dummy variables - - known as one-hot encoding of the data. 
        This process takes categorical variables, such as days of the week 
          and converts it to a numerical representation without an arbitrary ordering.'''
    dataframe = pd.get_dummies(dataframe, drop_first=True)  # To avoid dummy trap
    return dataframe

def create_X_y_datasets(df,target_column_name):
    '''create features and target datasets'''
    X = df[df.loc[:, df.columns != target_column_name].columns]
    y = df[target_column_name]
    return X, y  
    

#==========================================================
#Start of program - import data
#==========================================================

#import files for project - Load df from file
filename = 'S1_Validation_Loan_Basic_Data_Cleaning.csv'
df = pd.read_csv(filename)
print("\nLoaded dataframe.shape: ", df.shape)

#==========================================================
#baseline - Exploratory Data Analysis
#==========================================================

#find dataframe d.types
numeric_cols, non_numeric_cols = DF_data_types(df);pause()

# Exploratory Data - Descriptive
EDA_Descriptive(df);pause()

# Exploratory Data - Visualise
EDA_Visual(df);pause()

#==========================================================
#baseline - basic cleaning of data - defining consistent feature contents
#==========================================================

##is there missing data?
print(draw_missing_data_table(df));pause()

##What is the unique data?
dataframe_unique_check(df)

##Drop all duplicate rows based
print("\nNumber of rows before drop_duplicated: ",len(df))
df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
print("\nNumber of rows After drop_duplicated: ", len(df));pause()

##Lets re-examine the offending unique data of the dataframe after the duplicate row drops and the drop ID column?
check_columns = df[['ID', 'BAD_LOAN','LOAN_REASON','JOB']]
dataframe_unique_check(check_columns)
# BAD_LOAN: Expected TWO unique values but got FIVE. I will change to column data to Repaid, Defaulted.
# Then I will change to 1:Default and 0:Repaid as its affecting categorising
df['BAD_LOAN'].replace(['paid','Repaid'],0,inplace=True)
df['BAD_LOAN'].replace([ 'default', 'Dfault','Default'],1,inplace=True)

# LOAN_REASON: Expected TWO unique values but got SIX. will change to DebtCon, HomeImp
df['LOAN_REASON'].replace(['homeImp','Homeimp'],'HomeImp',inplace=True)
df['LOAN_REASON'].replace([ 'debtCon' , 'debtcon'],'DebtCon',inplace=True)
#Going to use this oppourtunity to impute a value other than leave empty
df['LOAN_REASON'].fillna('Other', inplace = True)

# Expected SIX unique values but got SEVEN. Will investigate empty features and call Other
# replacing na values in 'JOB' with 'Other'
df['JOB'].fillna('Other', inplace = True)

##Lets rexamine after replace function has been used
check_columns = df[['ID', 'BAD_LOAN','LOAN_REASON','JOB']]
dataframe_unique_check(check_columns)

#==========================================================
#Optimised cleaned data - In basic data clean step we imputed the mean for nan, lets that a closer look
#==================================================3========

#Lets look at the unique data to see if there any obovious fixes
dataframe_unique_check(df)

#Know lets see where they are missing
print(draw_missing_data_table(df))

# drop column not required
df.drop(['ID'], axis=1, inplace=True)

# Removing rows with many empty features

n = 8#1#8 #we are allowing rows with up to 7 empty cells
df = df[df.isnull().sum(axis=1) < n]

#==========================================================
# Optimised Data clean - Visual review of data and impute
#==========================================================

#Review the data for BAD_LOAN
column = 'BAD_LOAN'; plots_hist(column)

#Review the data for AMOUNT_REQUESTED
column = 'AMOUNT_REQUESTED'; plots_box(column);plots_hist(column)

#Review the data for LOAN_REASON
column = 'LOAN_REASON'; plots_hist(column)

##Impute columns missing data - less than 5%

# EXIST_PROPERTY_VALUE - approx 2% of data is missing going to apply the mean to the missing data
EXIST_PROPERTY_VALUE = round((df['EXIST_PROPERTY_VALUE'].mean()),1); df['EXIST_PROPERTY_VALUE'].fillna(EXIST_PROPERTY_VALUE, inplace=True)
column = 'EXIST_PROPERTY_VALUE'
plots_box(column);plots_hist(column)

# NO_OF_CR_LINES - approx 4% of data is missing going to apply the mean to the missing data
NO_OF_CR_LINES = round((df['NO_OF_CR_LINES'].mean()),1); df['NO_OF_CR_LINES'].fillna(NO_OF_CR_LINES, inplace=True)
column = 'NO_OF_CR_LINES'
plots_box(column);plots_hist(column)

# CR_LINES_AGE(MTS) - approx 5% of data missing so going to apply the mean to the missing data
CR_LINES_AGE = round((df['CR_LINES_AGE(MTS)'].mean()),1); df['CR_LINES_AGE(MTS)'].fillna(CR_LINES_AGE, inplace=True)
column = 'CR_LINES_AGE(MTS)'
plots_box(column);plots_hist(column)

##Impute columns missing data - less than 10%

# NO_OF_RECENT_CR_LINES - approx 9% of data missing so going to apply the mean to the missing data
NO_OF_RECENT_CR_LINES = round((df['NO_OF_RECENT_CR_LINES'].mean()),1); df['NO_OF_RECENT_CR_LINES'].fillna(NO_OF_RECENT_CR_LINES, inplace=True)
column = 'NO_OF_RECENT_CR_LINES'
plots_box(column);plots_hist(column)

# EMPLOYED_YEARS - approx 9% of data missing so going to apply the mean to the missing data
EMPLOYED_YEARS = round((df['EMPLOYED_YEARS'].mean()),1); df['EMPLOYED_YEARS'].fillna(EMPLOYED_YEARS, inplace=True)
column = 'EMPLOYED_YEARS'
plots_box(column);plots_hist(column)


# EXIST_MORTG_DEBT - approx 9% of data missing so going to apply the mean to the missing data
EXIST_MORTG_DEBT = round((df['EXIST_MORTG_DEBT'].mean()),1); df['EXIST_MORTG_DEBT'].fillna(EXIST_MORTG_DEBT, inplace=True)
column = 'EXIST_MORTG_DEBT'
plots_box(column);plots_hist(column)

# DELINQ_CR_LINES - approx 9% of data missing so going to apply the mean to the missing data
DELINQ_CR_LINES = round((df['DELINQ_CR_LINES'].mean()),1); df['DELINQ_CR_LINES'].fillna(DELINQ_CR_LINES, inplace=True)
column = 'DELINQ_CR_LINES'
plots_box(column);plots_hist(column)

##Impute columns missing data - approx 15% or less

# DEROG_REPORTS  - approx 12% of data missing so going to apply the mean to the missing data                0.119
DEROG_REPORTS = round((df['DEROG_REPORTS'].mean()),1); df['DEROG_REPORTS'].fillna(DEROG_REPORTS, inplace=True)
column = 'DEROG_REPORTS'
plots_box(column);plots_hist(column)

#==========================================================
#Sctter plot - lets examine the the relationship between a feature column and Target column
#==========================================================

#find dataframe d.types
numeric_cols, non_numeric_cols = DF_data_types(df)

#so before we impute lets examine the relation ship between the Target variable and Individual featuresusig a scatter plot
#Non-Numeric
List_of_Non_Numeric_Columns =  ['LOAN_REASON' , 'JOB']

#Numeric
list_of_Numeric_Columns =   ['BAD_LOAN', 'AMOUNT_REQUESTED', 'EXIST_MORTG_DEBT', 'EXIST_PROPERTY_VALUE',
                              'EMPLOYED_YEARS', 'DEROG_REPORTS', 'DELINQ_CR_LINES', 'CR_LINES_AGE(MTS)',
                               'NO_OF_RECENT_CR_LINES', 'NO_OF_CR_LINES', 'DEBT_TO_INCOME']

# iterate through list of columns with Numeric data     
for col in list_of_Numeric_Columns:
    plt.scatter(df[col], df['BAD_LOAN'], alpha=0.5)
    plt.title("Scatter plot of the relationship Bad_Loan to "+ str(col))
    plt.xlabel(str(col))
    plt.ylabel("Bad_Loan")    
    plt.show()

#==========================================================
#Feature engineering
#========================================================== 

##Impute columns missing data - approx 25% or less

from sklearn.impute import KNNImputer

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

df = transform_categorical_variables(df)

imputer = KNNImputer(n_neighbors=9)
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

## I just want to see is I can costruct a variable that looks at 

# LOAN	MORTDUE	VALUE
# 1100	25860	39025
# 1300	70053	68400
#BLoan risk
# Value > Mortgue and Loan < (value - mordue)


    
#==========================================================
#Checking for Multicollinearity
#========================================================== 

## https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc
## https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

# creating dummies for vif
df_cat = transform_categorical_variables(df)

# Create datasets for model
target_column_name = 'BAD_LOAN'
X, y = create_X_y_datasets(df_cat, target_column_name)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)for i in range(len(X.columns))]

# vif_data.drop(vif_data[0], axis=1, inplace = True)
vif_data.set_index('feature', inplace = True)
vif_data.sort_values(by=['VIF'], inplace=True, ascending=False)
print(vif_data)
print(vif_data.info())
vif_data.plot.bar()
plt.show()

#==========================================================
#Identiy important features
#==========================================================

#call function to plot and describe important features
most_important_features(X,y)

#Lets take a final look for missingness
print(draw_missing_data_table(df))

#==========================================================
#Save df to file
#==========================================================

filename = 'S7_Loan_Validation_Optimised_Data_Cleaning.csv'
df.to_csv(filename, index=False)
print("\n>>Saved df.shape: ", df.shape)