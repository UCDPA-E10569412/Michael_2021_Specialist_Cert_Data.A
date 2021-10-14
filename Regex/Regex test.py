# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:19:45 2021

@author: michael Impey
"""
import pandas as pd
import numpy as np
import re

#import files for project - Load df from file
filename = 'S1_Data_Gathering_Titanic.csv'
df = pd.read_csv(filename)
print("\nLoaded df.shape: ", df.shape);print("\ndf.info(): ",df.info())

#to convert all strings in "Name"to upper case just case there is a mix in the data
df['Name'] = df['Name'].str.upper()

# fill in missing ages using MR, Master, Mrs, Ms, Don, other. This is the use of REGEX for the Titanic data set to 
# get a mena age appropiate to the prefix.

#=============================================================================
# 1st we make a list fo prefixs
#=============================================================================

#find prefixs in Name column
prefix = [] # list to hold prefixes  
#iterate through rows of dataframe column 
for row in df['Name']:
    # strip after comma - start of string   
    row = re.sub(r'^.*,\s','',row) 
    # strip after full stop - end of string 
    row = re.sub(r'(?<=\.)[^.]*$','',row) 
    #append prefix to last prefix in list - prefix
    prefix.append(row)

#used for testing
# mylist = ['nowplaying', 'PBS', 'PBS', 'nowplaying', 'job', 'debate', 'thenandnow', 'impet', 'impet','impet']
# myset = set(mylist)
# print(myset)   

 
#set() -> new empty set object set(iterable) -> new set object
#Build an unordered collection of unique elements.
unique_prefix = set(prefix)
print(unique_prefix)

#output was:
    
#{'JONKHEER.', 'MAJOR.', 'THE COUNTESS.', 'MR.', 'MISS.', 'MRS. MARTIN (ELIZABETH L.', 
#'COL.', 'MME.', 'DR.', 'MS.', 'SIR.', 'LADY.', 'MASTER.', 'MRS.', 'CAPT.', 'REV.', 'DON.', 'MLLE.'}

# I am only selecting the following from the list for convienance - issue with would need to be resolved
list_of_prefixs = ('Mr','Mrs','Miss','Don','Master')

#=============================================================================
# 2nd  we have to find the title and make a colum where its True or false for each prefix
#=============================================================================

#create a new column in dataframe with prefixs - find rows in `df` which contain r'\,\sTEXT.\s' 
# df['Mr'] = (df[df['Name'].str.contains(r'\w,\sMR.\s\w')])# if upper() set
df['Mr'] = df['Name'].str.contains(r'\,\sMR.\s')
df['Mrs'] = df['Name'].str.contains(r'\,\sMRS')
df['Miss'] = df['Name'].str.contains(r'\,\sMISS.\s')
df['Don'] = df['Name'].str.contains(r'\,\sDON.\s')
df['Master'] = df['Name'].str.contains(r'\,\sMASTER\.\s')


#=============================================================================
#3rd interate through the list of prefixs and piut age in to the associated column prefix
#=============================================================================
for prefix in list_of_prefixs:
    # if iterate is the same as prefix do code
    if prefix == 'Mr':
        # find the column index no for prefix
        col_no = df.columns.get_loc(prefix)
        #f find the column number for Age
        col_no_age = df.columns.get_loc('Age')
        #using key and value take number for column with this prefix
        #interate through the prefix column rows and is the value is true do condition
        for index, value in df[prefix].items():
            if value == True:
                #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
                #assign the value in the age column at that location to the prefix column
                df.iloc[index, col_no] = df.iloc[index, col_no_age]
                
    if prefix == 'Mrs':
        col_no = df.columns.get_loc(prefix)
        col_no_age = df.columns.get_loc('Age')
        for index, value in df[prefix].items():
            if value == True:
                df.iloc[index, col_no] = df.iloc[index, col_no_age]
                
    if prefix == 'Miss':
        col_no = df.columns.get_loc(prefix)
        col_no_age = df.columns.get_loc('Age')
        for index, value in df[prefix].items():
            if value == True:
                df.iloc[index, col_no] = df.iloc[index, col_no_age]
                
    if prefix == 'Don':
        col_no = df.columns.get_loc(prefix)
        col_no_age = df.columns.get_loc('Age')
        for index, value in df[prefix].items():
            if value == True:
                df.iloc[index, col_no] = df.iloc[index, col_no_age]
                
    if prefix == 'Master':
        col_no = df.columns.get_loc(prefix)
        col_no_age = df.columns.get_loc('Age')
        for index, value in df[prefix].items():
            if value == True:
                df.iloc[index, col_no] = df.iloc[index, col_no_age]                

#=============================================================================
#4th Get mean of Ages in prefix column
#=============================================================================

age_mean = round((df["Age"].mean()), 1)
print("\nMean age on Titanic: ",age_mean)

#problem with nan skewing results - Get ages for prefixs
df['Mr_Age_Numeric'] = pd.to_numeric(df['Mr'], errors='coerce')
df['Mr_Age_Numeric'] = df['Mr_Age_Numeric'].replace(0, np.NaN)
imputed_age_Mr      = round((df['Mr_Age_Numeric'].mean()), 1)
print("imputed_age_Mr: ", (imputed_age_Mr))

df['Mrs_Age_Numeric'] = pd.to_numeric(df['Mrs'], errors='coerce')
df['Mrs_Age_Numeric'] = df['Mrs_Age_Numeric'].replace(0, np.NaN)
imputed_age_Mrs      = round((df['Mrs_Age_Numeric'].mean()), 1)
print("imputed_age_Mrs: ", (imputed_age_Mrs))

df['Miss_Age_Numeric'] = pd.to_numeric(df['Miss'], errors='coerce')
df['Miss_Age_Numeric'] = df['Miss'].replace(0, np.NaN)
imputed_age_Miss      = round((df['Miss_Age_Numeric'].mean()), 1)
print("imputed_age_Miss: ", (imputed_age_Miss))

df['Don_Age_Numeric'] = pd.to_numeric(df['Don'], errors='coerce')
df['Don_Age_Numeric'] = df['Don'].replace(0, np.NaN)
imputed_age_Don      = round((df['Don_Age_Numeric'].mean()),1)
print("imputed_age_Don: ", (imputed_age_Don))

df['Master_Age_Numeric'] = pd.to_numeric(df['Master'], errors='coerce')
df['Master_Age_Numeric'] = df['Master'].replace(0, np.NaN)
imputed_age_Master      = round((df['Master_Age_Numeric'].mean()), 1)
print("imputed_age_Master: ", (imputed_age_Master))

#output:
#    
# Mean age on Titanic:  29.7
# imputed_age_Mr:  32.4
# imputed_age_Mrs:  35.9
# imputed_age_Miss:  21.8
# imputed_age_Don:  40.0
# imputed_age_Master:  4.8

#=============================================================================
#5th Loop through rows, if df['Age'] is empty insert age depending on prefix
#=============================================================================

#get column locations for 
col_no_age = df.columns.get_loc('Age')
col_no_Name = df.columns.get_loc('Name')
col_no_ID = df.columns.get_loc('PassengerId')


#iterate through the dataframe, row by row
for i in range(len(df)):
    #if the age in the row is empty
    if str(df.iloc[i, col_no_age]) == str(np.nan):
        #assign True to each prefix and assin to variable
        df['Mr'] = df['Name'].str.contains(r'\,\sMR.\s')
        #assign prefix column location to variable
        col_no_mr = df.columns.get_loc('Mr')
        df['Mrs'] = df['Name'].str.contains(r'\,\sMRS.\s')
        col_no_mrs = df.columns.get_loc('Mrs')
        df['Miss'] = df['Name'].str.contains(r'\,\sMISS.\s')
        col_no_miss = df.columns.get_loc('Miss')
        df['Don'] = df['Name'].str.contains(r'\,\sDON.\s')
        col_no_don = df.columns.get_loc('Don')
        df['Master'] = df['Name'].str.contains(r'\,\sMASTER\.\s')
        col_no_master = df.columns.get_loc('Master')
        #if the prefix colum is True and the age was empty
        if df.iloc[i, col_no_mr] == True:
            print("1111111: ",  df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
            df.iloc[i, col_no_age] = imputed_age_Mr
            print("Changed to: ", df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
        elif df.iloc[i, col_no_mrs] == True:
            print("2222222: ",  df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age]) 
            df.iloc[i, col_no_age] = imputed_age_Mrs
            print("change to: ", df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
        elif df.iloc[i, col_no_miss] == True:
            print("3333333: ",  df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
            df.iloc[i, col_no_age] = imputed_age_Miss
            print("change to: ", df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
        elif df.iloc[i, col_no_don] == True:
            print("4444444: ",  df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
            df.iloc[i, col_no_age] = imputed_age_Don
            print("change to: ", df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
        elif df.iloc[i, col_no_master] == True:
            print("5555555: ",  df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
            df.iloc[i, col_no_age] = imputed_age_Master
            print("change to: ", df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
        else:
            #Catch any prefixs that are not above
            print("Other: ",  df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
            df.iloc[i, col_no_age] = age_mean
            print("change to: ", df.iloc[i, col_no_ID], df.iloc[i, col_no_Name], df.iloc[i, col_no_age])
        print()
        

#save to file so that we can see the changes
filename1 = 'S1_Data_Gathering_Regex_output.csv'
df.to_csv(filename1)
