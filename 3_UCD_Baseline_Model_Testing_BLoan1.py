# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 20:35:37 2021

@author: Michael Impey
"""
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
    models.append(RandomForestClassifier())
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
            print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[0][1])
            print("False positives - wrongly classified as Target: ", confusion_matrix_results[1][0])
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
                                                  'False_Neg':confusion_matrix_results[0][1],
                                                  'False_Pos':confusion_matrix_results[1][0],
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
# X = scale_data_normalisation(X)

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