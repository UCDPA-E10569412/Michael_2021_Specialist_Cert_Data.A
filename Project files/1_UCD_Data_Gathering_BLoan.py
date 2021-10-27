# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:35:37 2021

@author: Michael Impey
"""
import pandas as pd
import numpy as np

#===============================================================================
#Functions
#===============================================================================

def pause():
    '''used to pause program to view output'''
    input('===> Press Return to Continue Program ?')  

def import_file(filename):
    '''Import data - import and set up data frames'''
    file = pd.read_csv(filename)
    print("\n"+str(filename)+" in imported file:\n", file.info())
    return file

def create_project_file(A, B, Merge_on):
    '''create project file from imported files'''
    file = pd.merge(A, B, on=merge_on)
    print("\nMerged file info():\n", file.info())
    return file
     
#===============================================================================
#Start of program - import data
#===============================================================================

#import file for project
filename1 = 'loan_bad_ID_Target.csv'
df_a = import_file(filename1)

#import file for project
filename2 = 'loan_bad_ID_Features.csv'
df_b = import_file(filename2)

#create project file and merge on 'ID'
merge_on = "ID"
df_merge = create_project_file(df_a, df_b, merge_on)

#make a copy in case we ruin orignal dataframe
data = df_merge.copy()

#===============================================================================
#Tidy dataframe
#===============================================================================

#rename columns to more understanable titles
data.rename(columns={'BAD': 'BAD_LOAN', 
                   'LOAN': 'AMOUNT_REQUESTED',
                   'MORTDUE':'EXIST_MORTG_DEBT',
                   'VALUE': 'EXIST_PROPERTY_VALUE',
                   'REASON':'LOAN_REASON',
                   'YOJ': 'EMPLOYED_YEARS',
                   'DEROG': 'DEROG_REPORTS', 
                   'DELINQ': 'DELINQ_CR_LINES',
                   'CLAGE': 'CR_LINES_AGE(MTS)', 
                   'NINQ': 'NO_OF_RECENT_CR_LINES',
                   'CLNO': 'NO_OF_CR_LINES', 
                   'DEBTINC': 'DEBT_TO_INCOME'}, inplace=True)

#==========================================================
#Split dataFrame to Train and test and save
#==========================================================

#create a test and traing dataframe that has not been cleaned
#use the random function to select random rows and assign to a mask
msk = np.random.rand(len(data)) < 0.8

#save the test dataframe - not in mask
test = data[~msk]
filename1 = 'S1_Validation_Loan_Basic_Data_Cleaning.csv'
test.to_csv(filename1, index=False)
print("\n>>Saved test data.shape: ", test.shape);print(test.info())

#save the train dataframe in mask
train = data[msk]
filename2 = 'S1_Train_Test_Loan_Basic_Data_Cleaning.csv'
train.to_csv(filename2, index=False)
print("\n>>Saved train data.shape: ", train.shape);print(train.info())

#Load df from file - used to see how the saved file loads back as I had an issue with index column
df = pd.read_csv(filename2)
print("\n<<Loaded dataframe.shape: ", df.shape);print(df.info())#(4826, 14)