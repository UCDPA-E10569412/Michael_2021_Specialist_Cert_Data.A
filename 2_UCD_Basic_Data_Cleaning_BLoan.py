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
    return round(missing_data,3)

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
        print("\n>> Column "+str(col)+" Unique contents & count:\n",df[col].value_counts());pause()  
   
#==========================================================
#Start of program - import data
#==========================================================

#import files for project - Load df from file
filename = 'S1_train_Loan_Basic_Data_Cleaning.csv'
df = pd.read_csv(filename)
print("\nLoaded df.shape: ", df.shape)
#Loaded df.shape:  (5969, 14)

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
#baseline - basic cleaning of data - save for starting point in otimise data
#==========================================================

#Save basic cleaned data - this will be the starting point for Optimis_dat_cleaning
filename = 'S2_Loan_Basic_Data_Cleaning.csv'
df.to_csv(filename, index=False)
print("\n>>Saved df.shape: ", df.shape);print("\ndf.info(): ",df.info())

#==========================================================
#baseline - basic cleaning of data - Impute data for baseline model testing
#==========================================================

##Correct the missing data - first review
print(draw_missing_data_table(df));pause()

##Impute columns missing data - lets use a basic impute of the max value (by 10) in the column
for col in df.columns:
    print("column is "+str(col))
    n = df[col].max()
    df[col].fillna(n, inplace=True)
    print("Filled column["+str(col)+"] with "+str(n))
    
##Correct the missing data - review changes
print(draw_missing_data_table(df));pause()

#==========================================================
#Save df to file
#==========================================================

filename = 'S2_Loan_Basic_Data_for_Baseline_Models.csv'
df.to_csv(filename, index=False)
print("\n>>Saved dataframe shape: ", df.shape)
#>>Saved data frame shape:  (4768, 13)