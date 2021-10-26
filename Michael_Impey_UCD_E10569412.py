# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:35:37 2021

@author: Michael Impey
"""
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP1 - Data Gathering
#===
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================




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




#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP2 - Basic Data Cleaning
#===
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================





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
   
#==========================================================
#Start of program - import data
#==========================================================

#import files for project - Load df from file
filename = 'S1_Train_Test_Loan_Basic_Data_Cleaning.csv'
df = pd.read_csv(filename)
print("\nLoaded df.shape: ", df.shape)
#Loaded df.shape:  (5969, 14)

#==========================================================
#baseline - Exploratory Data Analysis
#==========================================================

#find dataframe d.types
numeric_cols, non_numeric_cols = DF_data_types(df)

# Exploratory Data - Descriptive
EDA_Descriptive(df)

# Exploratory Data - Visualise
EDA_Visual(df)

#==========================================================
#baseline - basic cleaning of data - defining consistent feature contents
#==========================================================

##is there missing data?
print(draw_missing_data_table(df))

##What is the unique data?
dataframe_unique_check(df)

##Drop all duplicate rows based
print("\nNumber of rows before drop_duplicated: ",len(df))
df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
print("\nNumber of rows After drop_duplicated: ", len(df))

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
print(draw_missing_data_table(df))

##Impute columns missing data - lets use a basic impute of the max value (by 10) in the column
for col in df.columns:
    print("column is "+str(col))
    n = df[col].max()
    n = n * 10
    df[col].fillna(n, inplace=True)
    print("Filled column["+str(col)+"] with "+str(n))
    
##Correct the missing data - review changes
print(draw_missing_data_table(df))

#==========================================================
#Save df to file
#==========================================================

filename = 'S2_Loan_Basic_Data_for_Baseline_Models.csv'
df.to_csv(filename, index=False)
print("\n>>Saved dataframe shape: ", df.shape)
#>>Saved data frame shape:  (4768, 13)


#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP3 - Baseline Model Testing
#===
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================



from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold 
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import mean
from numpy import isnan
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Functions
 
def pause(): 
    ''' used to pause program to view output '''
    input('===> Press Return to Continue Program ?')  
 
# get a list of models to evaluate
def get_models():
    models = list()
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())#
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    models.append(ExtraTreeClassifier())
    ##models.append(LinearSVC())# gives low reading but gives fault
    models.append(SVC())
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier())
    models.append(ExtraTreesClassifier())
    models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier())
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())   
    return models
 

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
 

def scale_data_normalisation(X):
    '''pre-processing - Normalisation'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled
    
def Classifier_models_test(df_model_values, a, b):
    '''Test data on a number of different classifier algorithims, using KFold CV and save performance data'''
    # get the list of models to consider
    models = get_models()
    
    # define test conditions
    Kfold_number = range(a,b,1)
    
    for CV_val in Kfold_number:
        #https://www.askpython.com/python/examples/k-fold-cross-validation
        kf = KFold(n_splits=CV_val, shuffle=True, random_state=42)
        # evaluate each model
        for model in models:
            print("\nKfold_number = ", CV_val)
            
            #Implementing cross validation and get y_p
            #https://www.bitdegree.org/learn/train-test-split
            import sklearn.model_selection as model_selection
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=CV_val)
            #Fit model and predict on training data 
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)        
            
            
            ##Test 1 - Accruacy_score
            #Implement accuracy_score() function          
            model_accuracy_score= accuracy_score(y_test, y_pred)
            print("\nAccuracy_score() is: ", round(model_accuracy_score, 3))

                
                
            ##Test 2 - Confusion matrix    
            confusion_matrix_results = confusion_matrix(y_test, y_pred)
            print("True negatives - correctly classified as not Target: ", confusion_matrix_results[0][0])
            print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[1][0])
            print("False positives - wrongly classified as Target: ", confusion_matrix_results[0][1])
            print("True positives - correctly classified as Target: " ,confusion_matrix_results[1][1])
            confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
            #just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
            assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
            print("Confusion Matric - Accuracy: " ,confusion_matric_accuracy)              

            
            ##Test 3 - Coss_Val_Score
            # evaluate model using each test condition on cross_val_score()
            #https://scikit-learn.org/stable/modules/cross_validation.html
            scores = cross_val_score(model,X,y,scoring='accuracy', cv=kf, n_jobs=None)
            cv_mean = mean(scores)
            # check for invalid results
            if isnan(cv_mean):
                continue
            # Model performances
            model_name = type(model).__name__
            print(str(model_name)+' Cross Val Score - Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

            
            #Append data to dataframe to record results
            df_model_values = df_model_values.append({'CV':CV_val,'Model':str(model_name),
                                                  'CVS_Accuracy':round((np.mean(scores)),3),
                                                  'CVS_STD':round((np.std(scores)),3), 
                                                  'Accuracy_Score':round(model_accuracy_score,3),
                                                  'C_M_Accuracy':round(confusion_matric_accuracy,2),
                                                  'True_Neg':confusion_matrix_results[0][0],
                                                  'False_Neg':confusion_matrix_results[1][0],
                                                  'False_Pos':confusion_matrix_results[0][1],
                                                  'True_Pos':confusion_matrix_results[1][1]},ignore_index = True)
    #Sort the values
    df_model_values.sort_values(by=['CVS_Accuracy'], axis=0, ascending=False,inplace=True, kind='quicksort',na_position='last',ignore_index=False, key=None)
    #save the dafatframe to file
    df_model_values.to_csv("ML3_Loans_Models_Results_on_Basic_Data.csv")#use this to see what the data looks like after lateststep
    return df_model_values
   
#==========================================================
#Start of program - import data
#==========================================================
#Load df from file
filename = 'S2_Loan_Basic_Data_for_Baseline_Models.csv'
df = pd.read_csv(filename)
print("\n<<Loaded Basic cleaned dataframe shape: ", df.shape);pause()
#(4768, 13

#==========================================================
#Pre-Processing - Check Classifier models on basic cleaned data
#==========================================================
#transform Categorical data to Numeric
df = transform_categorical_variables(df)

# Create datasets for model
target_column_name = 'BAD_LOAN'
X, y = create_X_y_datasets(df, target_column_name)

#rescale X between 0 - 1
X = scale_data_normalisation(X)

#check data shapes
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape);pause()

#Run All classifier model(s) test
#create a dataframe to capture model performance metrics
df_model_values = pd.DataFrame(data=None, columns = ['CV', 'Model', 'CVS_Accuracy', 'CVS_STD', 'Accuracy_Score',
                                                     'C_M_Accuracy','True_Neg','False_Neg','False_Pos',
                                                     'True_Pos'])
Kfold_start = 4
Kfold_Stop  = 11 #fold before thid intiger
df_model_values = Classifier_models_test(df_model_values, Kfold_start, Kfold_Stop)
print('\nSorted results for all models .head(20):\n',df_model_values.head(20))




#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP 4 - Optimise Data Cleaninig
#===
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================

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
    print("\nDataframe describe(all):\n", df.describe(include = 'all'))

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
filename = 'S2_Loan_Basic_Data_Cleaning.csv'
df = pd.read_csv(filename)
print("\nLoaded dataframe has shape: ", df.shape)

#==========================================================
#Optimised cleaned data - In basic data clean step we imputed the mean for nan, lets that a closer look
#==========================================================

#Lets look at the unique data to see if there any obovious fixes
dataframe_unique_check(df)

# drop column not required
df.drop(['ID'], axis=1, inplace=True)

#Know lets see where they are missing
print(draw_missing_data_table(df))

#==========================================================
# Removing rows with many empty features
#==========================================================

n = 8#1#8 #we are allowing rows with up to 7 empty cells
print("\ndataframe before dropping rows: ",df.info())
df = df[df.isnull().sum(axis=1) < n]
print("dataframe after dropping rows: ",df.info())

#==========================================================
# Optimised Data clean - Visual review of data and impute
#==========================================================

#Review the data for BAD_LOAN
column = 'BAD_LOAN'; plots_hist(column)

#Review the data for AMOUNT_REQUESTED
column = 'AMOUNT_REQUESTED'; plots_box(column);plots_hist(column)

#Review the data for LOAN_REASON
column = 'LOAN_REASON'; plots_hist(column)

#Review the data for LOAN_REASON
column = 'JOB'; plots_hist(column)

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
#replacing outlier with mean of column
df['EMPLOYED_YEARS'].replace(9999,EMPLOYED_YEARS,inplace=True)
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
#https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

from sklearn.impute import KNNImputer

df = transform_categorical_variables(df)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

imputer = KNNImputer(n_neighbors=4)
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

## I just want to see is I can costruct a variable that looks at 

# LOAN	MORTDUE	VALUE
# 1100	25860	39025
# 1300	70053	68400
#BLoan risk
# Value > Mortgue and Loan < (value - mordue)
column = "DEBT_TO_INCOME"
plots_box(column);plots_hist(column)

#Plot to see does KNN-Imputed values change scatter plot in any significant way
plt.scatter(df[column], df['BAD_LOAN'], alpha=0.5)
plt.title("Scatter plot of the relationship Bad_Loan to "+ str(col))
plt.xlabel(str(col))
plt.ylabel("Bad_Loan")    
plt.show()

#Know lets see where they are missing
print(draw_missing_data_table(df))
    
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

#==========================================================
#Save df to file
#==========================================================

filename = 'S4_Loan_Optimised_Data_Cleaning.csv'
df.to_csv(filename, index=False)
print("\n>>Saved df.shape: ", df.shape)








#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP 5 - Tune Model and Select Model
#===
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================







from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold 
from numpy import mean
from numpy import isnan
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#==========================================================
#Functions
#==========================================================

#used to pause program to view output  
def pause(): 
    input('===> Press Return to Continue Program ?') 
  
# get a list of models to evaluate
def get_models():
    models = list()
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())#
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    models.append(ExtraTreeClassifier())
    ##models.append(LinearSVC())# gives low reading but gives fault
    models.append(SVC())
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier())
    models.append(ExtraTreesClassifier())
    models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier())
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())   
    return models

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

def scale_data_standardisation(X):
    '''#pre-processing - Standarisation'''
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled
 
def scale_data_normalisation(X):
    '''#pre-processing - Normalisation'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled

def Classifier_models_test(df_model_values, a, b):
    '''Test data on a number of different classifier algorithims, using KFold CV and save performance data'''
    # get the list of models to consider
    models = get_models()
    
    # define test conditions
    Kfold_number = range(a,b,1)
    
    for CV_val in Kfold_number:
        #https://www.askpython.com/python/examples/k-fold-cross-validation
        kf = KFold(n_splits=CV_val, shuffle=True, random_state=42)
        # evaluate each model
        for model in models:
            print("\nKfold_number = ", CV_val)
            
            #Implementing cross validation and get y_p
            #https://www.bitdegree.org/learn/train-test-split
            import sklearn.model_selection as model_selection
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=CV_val)
            #Fit model and predict on training data 
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)        
            
            
            ##Test 1 - Accruacy_score
            #Implement accuracy_score() function          
            model_accuracy_score= accuracy_score(y_test, y_pred)
            print("\nAccuracy_score() is: ", round(model_accuracy_score, 3))

                
                
            ##Test 2 - Confusion matrix    
            confusion_matrix_results = confusion_matrix(y_test, y_pred)
            print("True negatives - correctly classified as not Target: ", confusion_matrix_results[0][0])
            print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[1][0])
            print("False positives - wrongly classified as Target: ", confusion_matrix_results[0][1])
            print("True positives - correctly classified as Target: " ,confusion_matrix_results[1][1])
            confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
            #just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
            assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
            print("Confusion Matric - Accuracy: " ,confusion_matric_accuracy)              

            
            ##Test 3 - Coss_Val_Score
            # evaluate model using each test condition on cross_val_score()
            #https://scikit-learn.org/stable/modules/cross_validation.html
            scores = cross_val_score(model,X,y,scoring='accuracy', cv=kf, n_jobs=None)
            cv_mean = mean(scores)
            # check for invalid results
            if isnan(cv_mean):
                continue
            # Model performances
            model_name = type(model).__name__
            print(str(model_name)+' Cross Val Score - Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

            
            #Append data to dataframe to record results
            df_model_values = df_model_values.append({'CV':CV_val,'Model':str(model_name),
                                                  'CVS_Accuracy':round((np.mean(scores)),3),
                                                  'CVS_STD':round((np.std(scores)),3), 
                                                  'Accuracy_Score':round(model_accuracy_score,3),
                                                  'C_M_Accuracy':round(confusion_matric_accuracy,2),
                                                  'True_Neg':confusion_matrix_results[0][0],
                                                  'False_Neg':confusion_matrix_results[1][0],
                                                  'False_Pos':confusion_matrix_results[0][1],
                                                  'True_Pos':confusion_matrix_results[1][1]},ignore_index = True)
    #Sort the values
    df_model_values.sort_values(by=['CVS_Accuracy'], axis=0, ascending=False,inplace=True, kind='quicksort',na_position='last',ignore_index=False, key=None)
    #save the dafatframe to file
    df_model_values.to_csv("ML5_Loans_Models_Results_on_Basic_Data_Scaled.csv")#use this to see what the data looks like after lateststep
    return df_model_values
   
#==========================================================
#Start of program - import data
#==========================================================

filename = 'S4_Loan_Optimised_Data_Cleaning.csv'
#Load df from file
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print("\ndf.info(): ",df.info())

#==========================================================
#Pre-Processing - Check Classifier models on basic cleaned data
#==========================================================

#transform Categorical data to Numeric
df = transform_categorical_variables(df)

# Create datasets for model
target_column_name = 'BAD_LOAN'
X, y = create_X_y_datasets(df, target_column_name)

#rescale X
# X = scale_data_normalisation(X)

#check data shapes
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape);pause()

#==========================================================
#Test classifier models
#==========================================================

#Run All classifier model(s) test
#create a dataframe to capture model performance metrics
df_model_values = pd.DataFrame(data=None, columns = ['CV', 'Model', 'CVS_Accuracy', 'CVS_STD', 'Accuracy_Score',
                                                     'C_M_Accuracy','True_Neg','False_Neg','False_Pos',
                                                     'True_Pos'])
Kfold_start = 4
Kfold_Stop  = 11 #fold before thid intiger
df_model_values = Classifier_models_test(df_model_values, Kfold_start, Kfold_Stop)
print('\nSorted results for all models .head(20):\n',df_model_values.head(20))








#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP6 - HyperTune Model
#===
#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================










# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:35:37 2021

@author: Michael Impey
"""
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

#==========================================================
#Functions
#==========================================================

#used to pause program to view output  
def pause(): 
    input('===> Press Return to Continue Program ?')  
    

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

def scale_data_standardisation(X):
    '''#pre-processing - Standarisation'''
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled
 
def scale_data_normalisation(X):
    '''#pre-processing - Normalisation'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled

#==========================================================
#Start of program - import data - Optimised Cleaned Data
#==========================================================
filename = 'S4_Loan_Optimised_Data_Cleaning.csv'
#Load df from file
df = pd.read_csv(filename)
print("\n<<Loaded dataframe shape: ", df.shape)

#==========================================================
#Pre-Processing
#==========================================================

#transform Categorical data to Numeric 
df = transform_categorical_variables(df)#giving a problem with ticket - text and numeric

# Create data set to train data 
target_column_name = 'BAD_LOAN'
# X = df[df.loc[:, df.columns != target_column_name].columns]
# y = df[target_column_name]

# Create datasets for model
X, y = create_X_y_datasets(df, target_column_name)

#rescale X
# X = scale_data_normalisation(X) # not sure if required for randonm forest

#Split data in to test and train - create cross validateing test and train data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

#check shape of ML data
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape)

#check shapes of Train and Test data
print("\nX_train.shape: ",X_train.shape)
print("y_train.shape: ", y_train.shape)
print("\nX_test.shape: ",X_test.shape)
print("y_test.shape: ", y_test.shape)

#==========================================================
#Hypertuning - Optimise final model 
#==========================================================


# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))
    
# model = ExtraTreesRegressor(**grid_result.best_params_)

# model.fit(train, y_train)



#Random Forest Classifier

print("\nHyperparameter Tuning Randon Forest:")    

param_grid = { "criterion" : ["gini", "entropy"], 
              "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
              "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], 
              "n_estimators": [100, 400, 700, 1000, 1500]}

# ##for quick testing as Hyptuning can take along time to complete
# param_grid = { "criterion" : ["gini", "entropy"], 
#               "min_samples_leaf" : [1], 
#               "min_samples_split" : [2], 
#               "n_estimators": [700]}

# Complete GRID search with various parameters to find best parameters
model = RandomForestClassifier( oob_score=True, random_state=1, n_jobs=-1)#n_estimators=100, max_features='auto',

clf = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=3)

clf.fit(X_train, y_train)

print("\nBest parameters found as per parama grid")
print("\nclf.best_params_", clf.best_params_) 

#=================================================
# Save Hypertuned parameters
#=================================================

# Import pickle Package
import pickle

# Save the Modle to file in the current working directory
model_filename = "6_Best_Model_Params.pkl"  

with open(model_filename, 'wb') as file:  
    pickle.dump(clf.best_params_, file)
    
#==========================================================
#Save model to file  # https://www.kaggle.com/prmohanty/python-how-to-save-and-load-ml-models
#==========================================================

# Save the Modle to file in the current working directory
model_filename = "6_Loan_UCD_ML_Model.pkl"  

with open(model_filename, 'wb') as file:  
    pickle.dump(clf, file)
       
#=================================================
# Perfom Classification Report after Hypertuning
#=================================================

from sklearn.metrics import classification_report   
print("\nDetailed classification report for HyperTuned model:")
print("Train scores:")
y_pred = clf.predict(X_train)
print(classification_report(y_train, y_pred))

print("Test scores:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#=================================================
# RandomForestClassifier set up parameters after Hypertuning
#=================================================

#Test new best paramters: Random Forest with TEST data
model = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 2,   
                                       n_estimators=700, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

##clf.best_params_ {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 700}
#Check model parameters loaded
print('\nConfirm parameters currently in use:\n')
pprint(model.get_params())

#get oob score
model.fit(X_train, y_train)
print("\nHypertuned - oob score:", round(model.oob_score_, 2)*100, "%")    

#Implement accuracy_score() function on model
y_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)
print("Accuracy_score() is: ", round(accuracy_test, 3))

#Confusion matrix    
confusion_matrix_results = confusion_matrix(y_test, y_pred)
print("True negatives - correctly classified as not Target: ", confusion_matrix_results[0][0])
print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[1][0])
print("False positives - wrongly classified as Target: ", confusion_matrix_results[0][1])
print("True positives - correctly classified as Target: " ,confusion_matrix_results[1][1])
confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
#just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
print("Confusion Matric - Accuracy: " ,confusion_matric_accuracy)              






#==============================================================================
#Extra TreeClassifier
#=============================================================================

print("\nHyperparameter Tuning Extra Trees:") 

# to identify the optimal parameters from this dictionary
# param_grid_ET={'criterion': ["gini", "entropy"],
#     'n_estimators': range(50,126,25),
#         'min_samples_leaf': range(1,30,2),
#         'min_samples_split': range(2,50,2)}


#for testing
param_grid_ET={'n_estimators': [50],
        'min_samples_leaf': [20],
        'min_samples_split': [15] }

model1 = ExtraTreesClassifier(random_state=1)#min_samples_split=25, min_samples_leaf=35, max_features=150
                            
gsc = GridSearchCV(estimator=model1, param_grid=param_grid_ET, n_jobs=-1, verbose=3)


Extra_trees_model = gsc.fit(X_train, y_train)

print("Best: %f using %s" % (Extra_trees_model.best_score_, Extra_trees_model.best_params_))


# #=================================================
# # Extra Trees Classification Report after Hypertuning
# #=================================================

from sklearn.metrics import classification_report   
print("\nDetailed confusion matrix for Extra trees HyperTuned model:")

#Test new best paramters: Random Forest with TEST data
model2 = ExtraTreesClassifier(criterion='gini', min_samples_leaf=1, min_samples_split=2, n_estimators=75)

#Check model parameters loaded
print('\nConfirm parameters currently in use:\n')
pprint(model2.get_params())

#get oob score
model2.fit(X_train, y_train)

#Implement accuracy_score() function on model
y_pred2 = model2.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred2)
print("Accuracy_score() is: ", round(accuracy_test, 2))

#Confusion matrix    
print("Confusion Matric - Accuracy: ")
print(confusion_matrix(y_test, y_pred2))




#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP7 - Optimise and Clean Validation Dataset
#===
#===============================================================================
#===============================================================================  
#===============================================================================
#===============================================================================





          

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











#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   STEP8 - Model predictions
#===
#===============================================================================
#===============================================================================  
#===============================================================================
#===============================================================================










import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from pprint import pprint
# from sklearn.ensemble import RandomForestClassifier

#==========================================================
#Functions
#==========================================================
#used to pause program to view output  
def pause(): 
    input('===> Press Return to Continue Program ?') 

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

def scale_data_standardisation(X):
    '''#pre-processing - Standarisation'''
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled
 
def scale_data_normalisation(X):
    '''#pre-processing - Normalisation'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled

#==========================================================
#load model from file 
#https://www.kaggle.com/prmohanty/python-how-to-save-and-load-ml-models
#==========================================================

# Model file name - Model from current working directory
model_filename = "6_Loan_UCD_ML_Model.pkl" 
# Load the Model back from file
model = pickle.load(open(model_filename, 'rb'))




    
#load best parameters from step 6 - GRID serach CV hypertuning
model_filename1 = "6_Best_Model_Params.pkl" 
# Load the Model back from file
with open(model_filename1, 'rb') as file:  
    best_params = pickle.load(file)

print(best_params)
# print("\nclf.best_params_", clf.best_params_)

#=================================================
# RandomForestClassifier parameters
#=================================================

#lets see parameters of model
print('\n1.Parameters currently in use:\n')
pprint(model.get_params())

#=================================================
#Re-load data to use for predictions - load the Validation data
#=================================================

#df = pd.read_csv("df_Data_for_saved_ML_data.csv",index_col=0)
filename = 'S7_Loan_Validation_Optimised_Data_Cleaning.csv'

#Load df from file
# dataFrame from Hyptune used on the model was df shape was : (4694, 18) there we need 18 columns
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print(df.info())#ratio 1(240):4(948)

#==========================================================
#Pre-Processing
#==========================================================

#transform Categorical data to Numeric 
df = transform_categorical_variables(df)#giving a problem with ticket - text and numeric

# Create data set to train data 
target_column_name = 'BAD_LOAN'
X = df[df.loc[:, df.columns != target_column_name].columns]
y = df[target_column_name]

# Create datasets for model
# X, y = create_X_y_datasets(df, target_column_name)

#rescale X
# X = scale_data_normalisation(X) # not sure if required for randonm forest

#check shape of ML data
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape)

#Make prediction (y_pred) of X
y_pred = model.predict(X)

#==========================================================
#model evaluatiuon
#==========================================================

# Models Accuracy
#Implement accuracy_score() function on model
accuracy_test = accuracy_score(y, y_pred)
print("Accuracy_score() is: ", round(accuracy_test, 3))

#Confusion Matrix:
from sklearn.metrics import confusion_matrix
confusion_matrix_results = confusion_matrix(y, y_pred) #  predictions)
# print("\nConfusion Matrix: \n",confusion_matrix_results)
print("\nConfusion Matrix: \nThe first row is about the not-target-predictions:")
print("True negatives - correctly classified as not Target: ", confusion_matrix_results[0][0])
print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[1][0])
print("\nThe second row is about the Target-predictions:")
print("False positives - wrongly classified as Target: ", confusion_matrix_results[0][1])
print("True positives - correctly classified as Target: " ,confusion_matrix_results[1][1])
confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
#just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
print("\nConfusion Matric - Accuracy: " ,confusion_matric_accuracy)  



#Precision and Recall:
from sklearn.metrics import precision_score, recall_score
print("\nPrecision and Recall results:")
print("Precision - predicts % of the time, a Target correctly:", round(precision_score(y, y_pred), 2))
print("Recall- tells us that it predicted the Target % of the correct:", round(recall_score(y, y_pred),2))
 

#Precision Recall Curve:
from sklearn.metrics import precision_recall_curve
# getting the probabilities of our predictions
precision, recall, threshold = precision_recall_curve(y, y_pred)
#function
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


#F-Score: 
from sklearn.metrics import f1_score
print("\nF1-score - combine precision and recall into one score")
print(f1_score(y, y_pred))



#Classification report
from sklearn.metrics import classification_report
print("Validation scores, Classification Report:")
print(classification_report(y, y_pred))


#ROC AUC Curve
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

#ROC AUC Score
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y, y_pred)
print("\nROC-AUC-Score - A classifiers that is 100% correct, \nwould have a ROC AUC Score of 1, score is :", r_a_score)










#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   Graphs for Project Report Insights
#===
#===============================================================================
#=============================================================================== 
#===============================================================================
#===============================================================================









# https://seaborn.pydata.org/tutorial/categorical.html
# Showing multiple relationships with facets

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="ticks", color_codes=True)

#df = pd.read_csv("df_Data_for_saved_ML_data.csv",index_col=0)
filename = 'S7_Loan_Validation_Optimised_Data_Cleaning.csv'

#Load df from file
# dataFrame from Hyptune used on the model was df shape was : (4694, 18) there we need 18 columns
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print(df.info())#ratio 1(240):4(948)

##create a list of columns
List_of_columns = []
for col in df.columns:
    List_of_columns.append(col)

##printlist so I can select what I want to review  
print(List_of_columns)

#Output
#['BAD_LOAN', 'AMOUNT_REQUESTED', 'EXIST_MORTG_DEBT', 'EXIST_PROPERTY_VALUE', 'EMPLOYED_YEARS', 
# 'DEROG_REPORTS', 'DELINQ_CR_LINES', 'CR_LINES_AGE(MTS)', 'NO_OF_RECENT_CR_LINES', 'NO_OF_CR_LINES', 
# 'DEBT_TO_INCOME', 'LOAN_REASON_HomeImp', 'LOAN_REASON_Other', 'JOB_Office', 'JOB_Other', 'JOB_ProfExe', 'JOB_Sales', 'JOB_Self'] 
 
#list of features o be examined 
Examine = ['BAD_LOAN']
#list of all fetaure i desired to do relationship against    
Examine2 =['BAD_LOAN', 'AMOUNT_REQUESTED', 'EXIST_MORTG_DEBT', 'EXIST_PROPERTY_VALUE', 'EMPLOYED_YEARS',
            'DEROG_REPORTS', 'DELINQ_CR_LINES', 'CR_LINES_AGE(MTS)', 'NO_OF_RECENT_CR_LINES', 'NO_OF_CR_LINES', 
            'DEBT_TO_INCOME', 'LOAN_REASON_HomeImp', 'LOAN_REASON_Other', 'JOB_Office', 'JOB_Other', 'JOB_ProfExe', 
            'JOB_Sales', 'JOB_Self'] 
 
# loop1 to go through list of features
for col in df[Examine]:
    
    #loop2 to go through list of features
    for col2 in df[Examine2]:
        
        #create a relationship plot
        sns.catplot(x=col, y=col2, hue="BAD_LOAN", data=df)
        #show plot
        plt.show()
        


sns.set_theme(style="white")


list_importance = ['BAD_LOAN','DEROG_REPORTS','NO_OF_CR_LINES','DELINQ_CR_LINES','DEBT_TO_INCOME','AMOUNT_REQUESTED', 'CR_LINES_AGE(MTS)','EXIST_PROPERTY_VALUE','EXIST_MORTG_DEBT']
# Compute the correlation matrix
corr = df[list_importance].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})








#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   REGEX example - uses Titanic
#===
#===============================================================================
#=============================================================================== 
#===============================================================================
#===============================================================================









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







#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   DEEP LEARNING - checking parameters of classifier model
#===
#===============================================================================
#=============================================================================== 
#===============================================================================
#===============================================================================








"""
Created on Thu Oct 21 19:32:52 2021
https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

https://datascience-george.medium.com/the-precision-recall-trade-off-aa295faba140

https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/

@author: micha
"""
# Import necessary modules
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
# Import EarlyStopping
from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import loadtxt # data = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv?_sm_au_=i0V0Nr7q4Npkwjwrp6JtKK7kjVH6W', delimiter=',')
from sklearn.preprocessing import MinMaxScaler
# Import the SGD optimizer
from keras.optimizers import SGD


#==========================================================
#Pre-Processing
#==========================================================

#used to pause program to view output  
def pause(): 
    input('===> Press Return to Continue Program ?') 

def  transform_categorical_variables(dataframe):
    ''' Transform categorical variables into dummy variables - - known as one-hot encoding of the data. 
        This process takes categorical variables, such as days of the week 
          and converts it to a numerical representation without an arbitrary ordering.'''
    dataframe = pd.get_dummies(dataframe, drop_first=True)  # To avoid dummy trap
    return dataframe

def scale_data_normalisation(X):
    '''#pre-processing - Normalisation'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled

def create_dataframe(df, filename):
    '''create data frame and pre-process it''' 
    print("\n"+str(filename)+": Is there empty cells in dataframe? "+str(df.isnull().any().any()))
    print(str(filename)+": Number of empty cells in datafrae? "+str(df.isnull().sum().sum()))
    #transform Categorical data to Numeric 
    df = transform_categorical_variables(df)#giving a problem with ticket - text and numeric    
    # Create data set to train data # split into input (X) and output (y) variables
    target_column_name = 'BAD_LOAN'
    X = df[df.loc[:, df.columns != target_column_name].columns]
    y = df[target_column_name]    
    #rescale X
    X = scale_data_normalisation(X) # not sure if required for randonm forest    
    #check shape of ML data
    print(str(filename)+" dataframe shape was :"+str(df.shape))
    print("X shape is :", X.shape)
    print("y shape is: ", y.shape)    
    #number of feature columns
    n_cols = int(X.shape[1])#option 1    
    input_shape = (n_cols,)#option 2
    print("Number of columns is: ",n_cols)
    print("The input shape is: ",input_shape)
    return X, y, n_cols

def Deep_Learniing_Model_Performance(loop_number, title0, title1, 
                                     X, y, X1, y1, Model_performance, 
                                     n_cols, Optimisers, hidden_layers, 
                                     hidden_layers1, hidden_layers2, batchs):    
    '''Test a number of model parameters'''
    #https://www.bitdegree.org/learn/train-test-split
    import sklearn.model_selection as model_selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=42)
            
    for hidden_layer in hidden_layers:
        
        for hidden_layer1 in hidden_layers1:
            
            for hidden_layer2 in hidden_layers2:
                
                for batch in batchs:
                    
                    for My_Optimiser in Optimisers:
       
                        print('\nLoop Number ['+str(loop_number)+']: \nOptimiser:'+str(My_Optimiser)+', hidden layers:'+str(hidden_layer)+\
                              ', hidden layers1:'+str(hidden_layer1)+', hidden layers2:'+str(hidden_layer2)+', Batch:'+str(batch))
     
                        #Create model
                        #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
                        #my_optimizer = 'sgd'#SGD(lr=lr)
                        model = Sequential()
                        model.add(Dense(hidden_layer, input_dim=n_cols, activation='relu'))
                        model.add(Dense(hidden_layer1, activation='relu'))
                        model.add(Dense(hidden_layer2, activation='relu'))
                        

                        # model.add(Dense(20, input_dim=n_cols, activation='relu'))
                        # model.add(Dense(20, activation='relu'))
                        # model.add(Dense(20, activation='relu'))
                        
                        model.add(Dense(1, activation='sigmoid'))
                        
                        #Compile Keras Model
                        model.compile(loss='binary_crossentropy', optimizer=My_Optimiser, metrics=['accuracy'])#optimizer=my_optimizer#sgd#adam
                        
                        #Fit Keras Model  #monitor='loss'
                        early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)##'loss'#'accuracy'#val_loss', mode='min', verbose=1)
                        
                        #fit Keras model with Training data
                        model.fit(X_train, y_train, epochs=1000, batch_size=batch, verbose=0, callbacks=[early_stopping_monitor]) #epochs=150
                        
                        #=================================
                        #Get prediction on Test data
                        #=================================                 
                        
                        # Evaluate Keras Model
                        y_pred = model.predict(X_test)
                        
                        #create list of predictions
                        y_pred = [round(x[0]) for x in y_pred]
                        
                        #Call accuracy model
                        Accuracy_model_Evaluate, Accuracy_Score = model_Accuracy_evaluation(y_test, y_pred, title0, model) 
                            
                        #Call confusion matirx model
                        FP, TN, FN, TP, confusion_matric_accuracy, confusion_matrix_results = model_ConfustionMatrix_evaluation(y_test, y_pred, title0)
                        
                        print(str(title0)+' Confusion Matrix');print(confusion_matrix_results)
                                               
                        #create Recals positive check, create negative check
                        P_ReCall = (TP/(TP+FN));print("\nP_ReCal:  ",P_ReCall)
                        N_ReCall = (TN/(TN+FP));print("N_ReCall: ",N_ReCall)
                        
                        model_ClassificationReport_evaluation(y_test, y_pred, title0)
                        
                        F1_score = model_F1Score_evaluation(y_test, y_pred, title0);print(F1_score)
                        

                        
                        #set variable for Validation equal to 0
                        TN1=FP1=FN1=TP1=P_ReCall1=N_ReCall1=F1_score1=0
                        
                        Validation_loop = False
                     
                        
                        
                        #good model predictability on Training data
                        if (P_ReCall > 0.5) and (N_ReCall > 0.5):
                            
                            print('\n'+str(title0)+' Results: \nOptimiser:'+str(My_Optimiser)+', hidden layers:'+str(hidden_layer)+\
                                   ', hidden layers1:'+str(hidden_layer1)+', hidden layers2:'+str(hidden_layer2)+', Batch:'+str(batch))
                             
                            #Print
                            print('\n'+str(title0)+' Model.Evaluate accuracy: '+str(Accuracy_model_Evaluate))
                            print(str(title0)+' Model accuracy_score: '+str(confusion_matric_accuracy))
                            print(str(title0)+' Model Accuracy Confusion matrix: '+str(Accuracy_Score))
                            print(str(title0)+' Confusion Matrix\n');print(confusion_matrix_results)
                            print('\nTN '+str(title0)+' : ' +str(TN))
                            print('FP '+str(title0)+' : ' +str(FP))
                            print('FN '+str(title0)+' : ' +str(FN))
                            print('TP '+str(title0)+' : ' +str(TP))
                            
                            #classification Report
                            model_ClassificationReport_evaluation(y_test, y_pred, title0)                            
                            
                            #=================================
                            #Get prediction on Validation data
                            #=================================
                            
                            # Evaluate Keras Model
                            y_pred1 = model.predict(X1)#X1 = X_test
                            
                            #create list of predictions
                            y_pred1 = [round(x[0]) for x in y_pred1]
                            
                            #Call accuracy model
                            Accuracy_model_Evaluate1, Accuracy_Score1 = model_Accuracy_evaluation(y1, y_pred1, title1, model)#y1=y_test
                            
                            #Call confusion matirx model
                            FP1, TN1, FN1, TP1, confusion_matric_accuracy1, confusion_matrix_results1 = model_ConfustionMatrix_evaluation(y1, y_pred1, title1)
                                                     
                            #create Recals positive check, create negative check
                            P_ReCall1 = (TP1/(TP1+FN1))
                            N_ReCall1 = (TN1/(TN1+FP1))
                            
                            print(P_ReCall1)
                            print(N_ReCall1) 
                            
                            
                           
                            #good model predictability on Validation data
                            if (P_ReCall1 > 0.6) and (N_ReCall1 > 0.6):
                                
                                Validation_loop = True
                                
                                #Row number
                                print('\n'+str(title1)+' Results: \nOptimiser:'+str(My_Optimiser)+', hidden layers:'+str(hidden_layer)+\
                                      ', hidden layers1:'+str(hidden_layer1)+', hidden layers2:'+str(hidden_layer2)+', Batch:'+str(batch))
                                
                                #Print evaluation metrics
                                print('\n'+str(title1)+' Model.Evaluate accuracy: '+str(Accuracy_model_Evaluate1))
                                print(str(title1)+' Model accuracy_score: '+str(confusion_matric_accuracy1))
                                print(str(title1)+' Model Accuracy Confusion matrix: '+str(Accuracy_Score1))
                                print(str(title1)+' Confusion Matrix\n');print(confusion_matrix_results1)
                                print('\nTN_ '+str(title1)+' : ' +str(TN1))
                                print('FP_ '+str(title1)+' : ' +str(FP1))
                                print('FN_ '+str(title1)+' : ' +str(FN1))
                                print('TP_ '+str(title1)+' : ' +str(TP1))
                                  
                                # model_PrecisionRecall_evaluation(y_test, y_pred1, title1)
                                
                                F1_score1 = model_F1Score_evaluation(y1, y_pred1, title1);print(F1_score1)
                                
                                #classification Report
                                model_ClassificationReport_evaluation(y1, y_pred1, title1)
                                
                                # model_ROC_AUC_evaluation(y_test, y_pred1, title1)

                        
                        #Append data to dataframe to record results
                        Model_performance = Model_performance.append({'Accuracy':Accuracy_model_Evaluate,
                                                                          'Optimiser':My_Optimiser, 
                                                                          'Hidden_Layer':hidden_layer, 
                                                                          'Hidden_Layer1':hidden_layer1,
                                                                          'Hidden_Layer2':hidden_layer2,
                                                                          'Batch': batch,
                                                                          'TN':TN, 'FP':FP,
                                                                          'FN':FN, 'TP':TP,
                                                                          'P_ReCall':P_ReCall, 'N_ReCall':N_ReCall,
                                                                          'F1_score':F1_score,
                                                                          'TN_':TN1, 'FP_':FP1,
                                                                          'FN_':FN1, 'TP_':TP1,
                                                                          'P_ReCall_':P_ReCall1, 'N_ReCall_':N_ReCall1,
                                                                          'F1_score_':F1_score1}, ignore_index = True)
                        
                        
                        
                        try:
                            #plot the relation ship between Positive recal and negative recal
                            scatter_plot_Recall_Pos_True_(loop_number, Model_performance['P_ReCall'], Model_performance['N_ReCall'], Model_performance['Optimiser'])
                        except:
                            print("issue with scatter plott: "+str(loop_number))
                        
                        
                        
                        #I only wnat this to plot when I have validation data
                        if Validation_loop == True:
                            try:
                                #plot the relation ship between Positive recal and negative recal
                                scatter_plot_Recall_Pos_True_(title1, Model_performance['P_ReCall_'], Model_performance['N_ReCall_'], Model_performance['Optimiser'])
                            except:
                                print("issue with scatter plott: "+str(title1))
                        

                        
                        loop_number = loop_number + 1
                        
                        Model_performance.to_csv("Deep_Learning_Loop_Results.csv")

def model_Accuracy_evaluation(y, y_pred, title, model):
    '''Accuracy score'''                  
    # https://datascience-george.medium.com/the-precision-recall-trade-off-aa295faba140
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9    
    from sklearn.metrics import accuracy_score
    #Implement accuracy_score() function on model
    accuracy_test = accuracy_score(y, y_pred)
    Accuracy_Score = round(accuracy_test, 3)    
    #get model accuracy
    _, accuracy = model.evaluate(X1, y1)
    Accuracy_model_Evaluate = (accuracy*100)
    
    return Accuracy_model_Evaluate, Accuracy_Score
 
    
def model_ConfustionMatrix_evaluation(y, y_pred, title):    
    '''Confusion Matrix'''    
    from sklearn.metrics import confusion_matrix    
    #create confusion_matric
    confusion_matrix_results = confusion_matrix(y, y_pred)    
    #confusion metrix variables                        
    FP = confusion_matrix_results[0][1]
    TN = confusion_matrix_results[0][0]
    FN = confusion_matrix_results[1][0]
    TP = confusion_matrix_results[1][1]    
    #just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
    assert len(y_pred)==(FP + TN + FN + TP)    
    confusion_matric_accuracy = (TN + TP)/len(y_pred)
    
    return FP, TN, FN, TP, confusion_matric_accuracy, confusion_matrix_results

    
def model_PrecisionRecall_evaluation(y, y_pred, title):
    print("Precision and Recalll for ", title)    
    '''Precision and Recall'''    
    from sklearn.metrics import precision_score, recall_score
    Precession = round(precision_score(y, y_pred), 2)
    Recall = round(recall_score(y, y_pred),2)
    #Precision Recall Curve:
    from sklearn.metrics import precision_recall_curve
    # getting the probabilities of our predictions
    precision, recall, threshold = precision_recall_curve(y, y_pred)
    #function
    def plot_precision_and_recall(precision, recall, threshold):
        plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
        plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
        plt.xlabel("threshold", fontsize=19)
        plt.legend(loc="upper right", fontsize=19)
        plt.ylim([0, 1])
    plt.figure(figsize=(14, 7))
    plot_precision_and_recall(precision, recall, threshold)
    plt.show()

    return Precession, Recall
   
    
def model_F1Score_evaluation(y, y_pred, title):
    '''F1 score'''
    from sklearn.metrics import f1_score
    F1_score = round(f1_score(y, y_pred),2)
    print(str(title)+", F1-score: "+str())
    
    return F1_score
 
    
def model_ClassificationReport_evaluation(y, y_pred, title):
    '''Classification report'''    
    from sklearn.metrics import classification_report
    print("\n"+str(title)+" - Classification Report")
    print(classification_report(y, y_pred))

    
def model_ROC_AUC_evaluation(y, y_pred, title):
    '''ROC AUC Curve'''
    print("ROC & AUC for ", title)        
    from sklearn.metrics import roc_curve
    # compute true positive rate and false positive rate
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
    # plotting them against each other
    def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
        plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'r', linewidth=4)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate (FPR)', fontsize=16)
        plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.figure(figsize=(14, 7))
    plot_roc_curve(false_positive_rate, true_positive_rate)
    plt.show()    
    #ROC AUC Score
    from sklearn.metrics import roc_auc_score
    r_a_score = roc_auc_score(y, y_pred)
    
    return r_a_score

def scatter_plot_Recall_Pos_True_(title, x_axis, y_axis, HUE):
    '''Plot the relationship bwteen the positive recall 
    and the negative recall for an elments'''
    colors = {'adam': 'gray', 'sgd': 'blue'}
    plt.figure(figsize=(14, 7))
    
    sns.scatterplot(x=x_axis, y=y_axis, hue = HUE,  palette=colors, s=50)
    
    plt.plot([0, 1], [0, 1], 'r', linewidth=1)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Positive Recall', fontsize=12)
    plt.ylabel('Negative Recall', fontsize=12)
    plt.title("Recall plots for different configerations of Optimisers :"+str(title))

    plt.show()

#==========================================================
#load training data frame
#==========================================================

#Load df from file
filename = 'S4_Loan_Optimised_Data_Cleaning.csv'
title0 = "Cleaned_Data"
df = pd.read_csv(filename)

#create dataframe
X, y, n_cols = create_dataframe(df, filename)

#==========================================================
#Load validation dataframe
#==========================================================

#Load df from file
# filename = 'S4_Loan_Optimised_Data_Cleaning.csv'
filename = 'S7_Loan_Validation_Optimised_Data_Cleaning.csv'
title1 = "Validation_Data"
df = pd.read_csv(filename)

#create dataframe
X1, y1, n_cols1 = create_dataframe(df, filename)


#==========================================================
#set up parameters
#==========================================================

loop_number = 1

#https://adatis.co.uk/introduction-to-artificial-neural-networks/
#https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
#https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
lr_to_test = [0.01]#[0.000001, 0.0001,  0.01, 1]
Optimisers = ['adam','sgd']
hidden_layers  = range(10, 30, 2)#[20]#range(100, 1, -20)#[128]#range(50, 30, -1)
hidden_layers1 = range(5, 30, 2)#[20]#range(100, 1, -20)#[64]#range(40, 20, -1)
hidden_layers2 = range(3, 30, 2)#[32]#range(30, 10, -1)#20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]

#The batch size is a number of samples processed before the model is updated. #The number of epochs is the number of complete passes through the training dataset. 
#The size of a batch must be more than or equal to one #and less than or equal to the number of samples in the training dataset
#https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/#:~:text=The%20batch%20size%20is%20a%20number%20of%20samples%20processed%20before,samples%20in%20the%20training%20dataset.
batchs = [10]#range(8, 12, 1)

#create a dataframe to capture model performance metrics
Model_performance = pd.DataFrame(data=None, columns = ['Accuracy','Optimiser', 'Hidden_Layer','Hidden_Layer1','Hidden_Layer2', 'Batch'])

#Call program function
Deep_Learniing_Model_Performance(loop_number, title0, title1, 
                                     X, y, X1, y1, Model_performance, 
                                     n_cols, Optimisers, hidden_layers, 
                                     hidden_layers1, hidden_layers2, batchs)




#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                   DEEP LEARNING - Running parameters in classifier model
#===
#===============================================================================
#=============================================================================== 
#===============================================================================
#===============================================================================



# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 18:48:57 2021

@author: michael Impey

"""

# Import necessary modules
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
# Import EarlyStopping
from keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import loadtxt # data = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv?_sm_au_=i0V0Nr7q4Npkwjwrp6JtKK7kjVH6W', delimiter=',')
from sklearn.preprocessing import MinMaxScaler
# Import the SGD optimizer
from keras.optimizers import SGD

#==========================================================
#Pre-Processing
#==========================================================

#used to pause program to view output  
def pause(): 
    input('===> Press Return to Continue Program ?') 

def  transform_categorical_variables(dataframe):
    ''' Transform categorical variables into dummy variables - - known as one-hot encoding of the data. 
        This process takes categorical variables, such as days of the week 
          and converts it to a numerical representation without an arbitrary ordering.'''
    dataframe = pd.get_dummies(dataframe, drop_first=True)  # To avoid dummy trap
    return dataframe

def scale_data_normalisation(X):
    '''#pre-processing - Normalisation'''
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled

def scatter_plot_Recall_Pos_True_(title, x_axis, y_axis, HUE):
    '''Plot the relationship bwteen the positive recall 
    and the negative recall for an elments'''
    colors = {'adam': 'gray', 'sgd': 'blue'}
    plt.figure(figsize=(14, 7))
    
    sns.scatterplot(x=x_axis, y=y_axis, hue = HUE,  palette=colors, s=50)
    
    plt.plot([0, 1], [0, 1], 'r', linewidth=1)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Positive Recall', fontsize=12)
    plt.ylabel('Negative Recall', fontsize=12)
    plt.title("Recall plots for different configerations of Optimisers :"+str(title))

    plt.show()

#=================================================
#Re-load data to use for predictions - load the Train, test data
#=================================================

#df = pd.read_csv("df_Data_for_saved_ML_data.csv",index_col=0)
filename = 'S4_Loan_Optimised_Data_Cleaning.csv'

#Load df from file
# dataFrame from Hyptune used on the model was df shape was : (4694, 18) there we need 18 columns
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print(df.info())#ratio 1(240):4(948)

print("Is there empty cells in dataframe: ",df.isnull().any().any())
print("Number of empty cells in datafrae: ",df.isnull().sum().sum())

#==========================================================
#Pre-Processing
#==========================================================

#1. load and define data
#transform Categorical data to Numeric 
df = transform_categorical_variables(df)#giving a problem with ticket - text and numeric

# Create data set to train data 
# split into input (X) and output (y) variables
target_column_name = 'BAD_LOAN'
X = df[df.loc[:, df.columns != target_column_name].columns]
y = df[target_column_name]

#rescale X
X = scale_data_normalisation(X) # not sure if required for randonm forest

#check shape of ML data
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape)

#2. Define Keras Model
#number of feature columns
n_cols = int(X.shape[1])#option 1

input_shape = (n_cols,)#option 2
print(n_cols)
print(input_shape)

#Fully connected layers are defined using the Dense class. We can specify the number of neurons or nodes in the layer as the first argument, and specify the activation function using the activation argument.
model = Sequential()
# The model expects rows of data with 8 variables (the input_dim=8 argument) /The first hidden layer has 12 nodes and uses the relu activation function.
model.add(Dense(26, input_dim=n_cols, activation='relu'))#input_dim=8
# #The second hidden layer has 8 nodes and uses the relu activation function.
model.add(Dense(21, activation='relu'))
model.add(Dense(19, activation='relu'))


#The output layer has one node and uses the sigmoid activation function.
# We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.
model.add(Dense(1, activation='sigmoid'))#model.add(Dense(2, activation='softmax'))
 
#3. Compile Keras Model
#Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.
#training a network means finding the best set of weights to map inputs to outputs in our dataset
#cross entropy as the loss argument. This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“
#define the optimizer as the efficient stochastic gradient descent algorithm “adam“.
#because it is a classification problem, we will collect and report the classification accuracy, defined via the metrics argument.
# compile the keras model
#https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008
my_otimizer  = 'sgd'#adam'#SGD(lr=0.000001)
model.compile(loss='binary_crossentropy', optimizer=my_otimizer, metrics=['accuracy'])#sgd#adam

#4. Fit Keras Model
# Define early_stopping_monitor
#https://keras.io/api/callbacks/early_stopping/ #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
early_stopping_monitor = EarlyStopping(monitor='loss', patience=5)##'loss'#'accuracy'#val_loss', mode='min', verbose=1)
# early_stopping_monitor = EarlyStopping(patience=2)

#execute the model on some data
#Epoch: One pass through all of the rows in the training dataset. The training process will run for a fixed number of iterations through the dataset called epochs
#Batch: One or more samples considered by the model within an epoch before weights are updated. We must also set the number of dataset rows that are considered before the model weights are updated within each epoch, called the batch size and set using the batch_size argument.
# fit the keras model on the dataset
model.fit(X, y, epochs=1000, batch_size=10, callbacks=[early_stopping_monitor]) #epochs=150

#5. Evaluate Keras Model

#Make prediction (y_pred) of X
y_pred = model.predict(X)

# round predictions
y_pred = [round(x[0]) for x in y_pred]

# evaluate the keras model on test train data
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100)) # loss has been left out - #we would like the loss to go to zero and accuracy to go to 1.0 (e.g. 100%)

#Confusion Matrix:
from sklearn.metrics import confusion_matrix
confusion_matrix_results = confusion_matrix(y, y_pred) #  predictions)
print("TN : ", confusion_matrix_results[0][0])
print("FP : ", confusion_matrix_results[0][1])
print("FN : ", confusion_matrix_results[1][0])
print("TP : " ,confusion_matrix_results[1][1]) 
confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
#just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
print("\nConfusion Matric - Accuracy: " ,confusion_matric_accuracy)  





pause()


#6. Make Predictions on validation data

#=================================================
#Re-load data to use for predictions - load the Validation data
#=================================================

#df = pd.read_csv("df_Data_for_saved_ML_data.csv",index_col=0)
filename = 'S7_Loan_Validation_Optimised_Data_Cleaning.csv'

# dataFrame from Hyptune used on the model was df shape was : (4694, 18) there we need 18 columns
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print(df.info())#ratio 1(240):4(948)

print("Is there empty cells in dataframe: ",df.isnull().any().any())
print("Number of empty cells in datafrae: ",df.isnull().sum().sum())

#==========================================================
#Pre-Processing prediction
#==========================================================

#transform Categorical data to Numeric 
df = transform_categorical_variables(df)#giving a problem with ticket - text and numeric
df.to_csv("testcat,csv")
# Create data set to train data 
target_column_name = 'BAD_LOAN'
X = df[df.loc[:, df.columns != target_column_name].columns]
y = df[target_column_name]

# Create datasets for model
# X, y = create_X_y_datasets(df, target_column_name)

#rescale X
X = scale_data_normalisation(X) # not sure if required for randonm forest

#check shape of ML data
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape)

#Make prediction (y_pred) of X
y_pred = model.predict(X)

# round predictions
y_pred = [round(x[0]) for x in y_pred]

#==========================================================
#model evaluatiuon
#==========================================================

# Models Accuracy
from sklearn.metrics import accuracy_score
#Implement accuracy_score() function on model
accuracy_test = accuracy_score(y, y_pred)
print("Accuracy_score() is: ", round(accuracy_test, 3))

#Confusion Matrix:
from sklearn.metrics import confusion_matrix
confusion_matrix_results = confusion_matrix(y, y_pred) #  predictions)
print("TN : ", confusion_matrix_results[0][0])
print("FP : ", confusion_matrix_results[0][1])
print("FN : ", confusion_matrix_results[1][0])
print("TP : " ,confusion_matrix_results[1][1]) 
confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
#just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
print("\nConfusion Matric - Accuracy: " ,confusion_matric_accuracy)  

#Precision and Recall:
from sklearn.metrics import precision_score, recall_score
print("\nPrecision and Recall results:")
print("Precision - predicts % of the time, a Target correctly:", round(precision_score(y, y_pred), 2))
print("Recall- tells us that it predicted the Target % of the correct:", round(recall_score(y, y_pred),2))
 
#Precision Recall Curve:
from sklearn.metrics import precision_recall_curve
# getting the probabilities of our predictions
precision, recall, threshold = precision_recall_curve(y, y_pred)
#function
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

#F-Score: 
from sklearn.metrics import f1_score
print("\nF1-score - combine precision and recall into one score")
print(f1_score(y, y_pred))

#Classification report
from sklearn.metrics import classification_report
print("Validation scores, Classification Report:")
print(classification_report(y, y_pred))

#ROC AUC Curve
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

#ROC AUC Score
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y, y_pred)
print("\nROC-AUC-Score - A classifiers that is 100% correct, \nwould have a ROC AUC Score of 1, score is :", r_a_score)



#===============================================================================
#===============================================================================
#===============================================================================
#===============================================================================
#===
#===                  Feature Importance
#===
#===============================================================================
#=============================================================================== 
#===============================================================================
#===============================================================================




# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:18:39 2021

@author: micha


https://colab.research.google.com/drive/1as0n3ozs4ut7-KbQX-d1NVeP37-E6kkC#scrollTo=mEp0fCnZPoMr

https://medium.com/bigdatarepublic/feature-importance-whats-in-a-name-79532e59eea3

"""

#Install non-standard packages (assuming jupyter notebook)
# !pip install shap
# !pip install lime
# !pip install eli5
# pip install -U scikit-learn scipy matplotlib

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

#scikit-learn package (https://pypi.org/project/scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
# from sklearn.metrics.scorer import make_scorer#having issue with this import - cannot find
from sklearn.metrics import fbeta_score, make_scorer

#eli5 package (https://eli5.readthedocs.io/en/latest)
import eli5
from eli5.sklearn import PermutationImportance

#lime package (https://github.com/marcotcr/lime)
import lime
import lime.lime_tabular

#shap package (https://github.com/slundberg/shap)
import shap

##Load data and train model




def DF_data_types(df):
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
    return numeric_cols, non_numeric_cols 



import pandas as pd
#import files for project - Load df from file
filename = 'plot_permutation_importance.csv'
filename = 'S4_Loan_Optimised_Data_Cleaning.csv'
df = pd.read_csv(filename)
print("\nLoaded df.shape: ", df.shape);print("\ndf.info(): ",df.info())


numeric_cols, non_numeric_cols = DF_data_types(df)




# Create data set to train data 
target_column_name = 'BAD_LOAN'#'Survived'
X = df[df.loc[:, df.columns != target_column_name].columns]
y = df[target_column_name]



#Split train and test set.
RANDOM_STATE = 123
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

#Train random forest classification model
model = RandomForestClassifier(max_depth=4, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Diagnosis prediction
y_predict = model.predict(X_test)

# Probability of malignant tissue produced by the model
y_prob = [probs[1] for probs in model.predict_proba(X_test)]
# y_prob = [round(probs[1]) for probs in model.predict_proba(X_test)]#affects ROC curve

print(y_prob)

input("hjkgh")


##Evaluate model

#Accuracy on test set
print(f"Test accuracy: {accuracy_score(y_test, y_predict).round(2)}")

# Confusion matrix test set
pd.DataFrame(
    confusion_matrix(y_test, y_predict)
)

# Compute area under the curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

#Set default figure size
plt.rcParams['figure.figsize'] = (8,8)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Bank loans")
plt.legend(loc="lower right")
plt.show()

##Model-specific feature importance

# Feature importance dataframe
imp_df = pd.DataFrame({'feature': X_train.columns.values,
                       'importance': model.feature_importances_})
 
# Reorder by importance
ordered_df = imp_df.sort_values(by='importance')
imp_range=range(1,len(imp_df.index)+1)
 
## Barplot with confidence intervals
height = ordered_df['importance']
bars = ordered_df['feature']
y_pos = np.arange(len(bars))

# Create horizontal bars
plt.barh(y_pos, height)
 
# Create names on the y-axis
plt.yticks(y_pos, bars)

plt.xlabel("Mean reduction in tree impurity in random forest")

plt.tight_layout()
# Show graphic
plt.show()

##Permutation feature importance

# Feature importance based on TRAINING set

perm_test = PermutationImportance(model, scoring=make_scorer(roc_auc_score), n_iter=50, random_state=RANDOM_STATE, cv="prefit")

# fit and see the permuation importances
perm_test.fit(X_train, y_train)

imp_df = eli5.explain_weights_df(perm_test)
label_df = pd.DataFrame({'feature': [ "x" + str(i) for i in range(len(X_test.columns))], 'feature_name': X_test.columns.values})
imp_df = pd.merge(label_df, imp_df, on='feature', how='inner', validate="one_to_one")
 
# Reorder by importance
ordered_df = imp_df.sort_values(by='weight')
imp_range=range(1,len(imp_df.index)+1)
 

## Barplot with confidence intervals

height = ordered_df['weight']
bars = ordered_df['feature_name']
ci = 1.96 * ordered_df['std']
y_pos = np.arange(len(bars))

# Create horizontal bars
plt.barh(y_pos, height, xerr=ci)
 
# Create names on the y-axis
plt.yticks(y_pos, bars)

plt.xlabel("Permutation feature importance training set (decrease in AUC)")
plt.tight_layout()

# Show graphic
plt.show()

## LIME
input("ddkbndkjbnd")
#Explain samples in test set
X_explain = X_test



explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values,
                                                   feature_names=X_train.columns.values,
                                                   discretize_continuous=True,
                                                   mode="classification",
                                                   verbose=True,
                                                   random_state=RANDOM_STATE)

#Explaining first subject in test set using all 30 features
exp = explainer.explain_instance(X_explain.values[0,:],model.predict_proba, num_features=30)
#Plot local explanation
plt = exp.as_pyplot_figure()
plt.tight_layout()
exp.show_in_notebook(show_table=True)

# explain the model's predictions on test set using SHAP values
# same syntax works for xgboost, LightGBM, CatBoost, and some scikit-learn models
explainer = shap.TreeExplainer(model)

# shap_values consists of a list of two matrices of dimension samplesize x #features
# The first matrix uses average nr of benign samples as base value
# The second matrix which is used below uses average nr of malignant samples as base value 
shap_values = explainer.shap_values(X_explain)


# Interactive visualization of the explanation of the first subject
# in the test set (X_explain).
# It shows the relative contribution of features to get from the base value of malignant
# samples(average value)
# to the output value (1 in case of malignant sample)
# the numbers at the bottom show the actual values for this sample.
shap.initjs() #initialize javascript in cell
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_explain.iloc[0,:])


#Interactive visualization of all sample/feature Shapley values
#It is possible to show the relative contribution of individual features for all
# samples on the y-axis as well.
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_explain)



#A summary plot with the shapley value (feature importance) 
shap.summary_plot(shap_values[1], X_explain)



#Same as above, but with violin plots to better see the distribution of shapley values 
shap.summary_plot(shap_values[1], X_explain, plot_type="violin")





