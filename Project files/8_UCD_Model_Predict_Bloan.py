# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:04:26 2021

@author: micha
"""
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
print("\nConfusion Matrix: \n",confusion_matrix_results)
print("\nConfusion Matrix: \nThe first row is about the not-target-predictions:")
print("True negatives - correctly classified as not Target: ", confusion_matrix_results[0][0])
print("False positives - wrongly classified as Target: ", confusion_matrix_results[0][1])

print("\nThe second row is about the Target-predictions:")
print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[1][0])
print("True positives - correctly classified as Target: " ,confusion_matrix_results[1][1])
confusion_matric_accuracy = (confusion_matrix_results[0][0]+confusion_matrix_results[1][1])/len(y_pred)
#just want to make sure program stops if these couts are dirrenent as it mean my accuracy will not be correct
assert len(y_pred)==(confusion_matrix_results[0][0]+confusion_matrix_results[0][1]+confusion_matrix_results[1][0]+confusion_matrix_results[1][1])
print("\nConfusion Matric - Accuracy: " ,confusion_matric_accuracy)  



#Precision and Recall:
from sklearn.metrics import precision_score, recall_score
print("\nPrecision and Recall results:")
print("Precision - predicts % of the time, a Target correctly:", round(precision_score(y, y_pred), 2))
print("Recall- tells us that it predicted the Target % correclyt:", round(recall_score(y, y_pred),2))
 

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