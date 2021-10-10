# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:35:37 2021

@author: Michael Impey
"""
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
    # get the list of models to consider
    models = get_models()
    # define test conditions
    Kfold_number = range(a,b,1)
    for CV_val in Kfold_number:
        cv = KFold(n_splits=CV_val, shuffle=True, random_state=42)
        # evaluate each model
        for model in models:
            print("\nKfold_number = ", CV_val)
            # evaluate model using each test condition on cross_val_score()
            scores = cross_val_score(model,X,y,scoring='accuracy', cv=cv, n_jobs=None)
            cv_mean = mean(scores)
            # check for invalid results
            if isnan(cv_mean):
                continue
            
            # Model performances
            model_name = type(model).__name__
            print(str(model_name)+' CV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
            
            #Implement accuracy_score() function
            model.fit(X,y)
            y_pred = model.predict(X)
            #Accuracy score on X
            try:
                accuracy_train = accuracy_score(y, y_pred)
                print("Accuracy_score() is: ", round(accuracy_train, 3))
            except:
                print("accuracy score is void")
                accuracy_train = 999
            df_model_values = df_model_values.append({'CV':CV_val,'Model':str(model_name),'Model_Accuracy':round((np.mean(scores)),3),
                                                      'Model_STD':round((np.std(scores)),3), 'Accuracy_Score': round(accuracy_train,3)}, ignore_index = True)
    df_model_values.sort_values(by=['Model_Accuracy'], axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last', ignore_index=False, key=None)
    df_model_values.to_csv("ML5_Loans_Models_Results_on_Optimised_Data.csv")#use this to see what the data looks like after lateststep
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
X = scale_data_normalisation(X)

#check data shapes
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape);pause()

#==========================================================
#Test classifier models
#==========================================================

#create a dataframe to capture model performance metrics
df_model_values = pd.DataFrame(data=None, columns = ['CV', 'Model', 'Model_Accuracy', 'Model_STD', 'Accuracy_Score'])
Kfold_start = 4
Kfold_Stop  = 11 #fold before thid intiger
df_model_values = Classifier_models_test(df_model_values, Kfold_start, Kfold_Stop)
print('\nSorted results for all models .head(20):\n',df_model_values.head(20))

