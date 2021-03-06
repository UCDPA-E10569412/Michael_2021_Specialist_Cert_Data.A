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

# {'bootstrap': False,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 'auto',
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_impurity_split': None,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 75,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': None,
#  'verbose': 0,
#  'warm_start': False}

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

 
