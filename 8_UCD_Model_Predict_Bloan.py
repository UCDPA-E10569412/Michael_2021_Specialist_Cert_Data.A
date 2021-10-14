# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:04:26 2021

@author: micha
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier

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

# Model file name - Modle from current working directory
model_filename = "6_Loan_UCD_ML_Model.pkl" 

# Load the Model back from file
with open(model_filename, 'rb') as file:  
    model = pickle.load(file)
    
#load best parameters from step 6 - GRID serach CV hypertuning
model_filename = "6_Best_Model_Params.pkl" 

# Load the Model back from file
with open(model_filename, 'rb') as file:  
    best_params = pickle.load(file)

print(best_params);pause()
# print("\nclf.best_params_", clf.best_params_) ;pause()

#=================================================
# RandomForestClassifier parameters
#=================================================

#lets see parameters of model
print('\n1.Parameters currently in use:\n')
pprint(model.get_params());pause()

#=================================================
#Re-load data to use for predictions - load the TEST data
#=================================================

#df = pd.read_csv("df_Data_for_saved_ML_data.csv",index_col=0)
filename = 'S7_Loan_Optimised_Data_Cleaning.csv'
#Load df from file
# dataFrame from Hyptune used on the model was df shape was : (4694, 18) there we need 18 columns
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print(df.info());pause()

#==========================================================
#Pre-Processing
#==========================================================

# Create data set to train data 
target_column_name = 'BAD_LOAN'
X = df[df.loc[:, df.columns != target_column_name].columns]
y = df[target_column_name]

#transform Categorical data to Numeric 
df = transform_categorical_variables(df)#giving a problem with ticket - text and numeric

# Create datasets for model
X, y = create_X_y_datasets(df, target_column_name)

#rescale X
# X = scale_data_normalisation(X) # not sure if required for randonm forest

#check shape of ML data
print("\ndf shape was :", df.shape)
print("X shape is :", X.shape)
print("y shape is: ", y.shape)

# #Split data in to test and train - create cross validateing test and train data sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

# #check shapes of Train and Test data
# print("\nX_train.shape: ",X_train.shape)
# print("y_train.shape: ", y_train.shape)
# print("\nX_test.shape: ",X_test.shape)
# print("y_test.shape: ", y_test.shape);pause()

#Thsi is so we make the predictions against the whole data set 
X_train = X;X_test = X;y_train = y;y_test = y



#==========================================================
# Reload model using araenters : if you would like to reload model using best parameters
#==========================================================

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
print("Accuracy_score() is: ", round(accuracy_test, 3));pause()

#==========================================================
#model evaluatiuon
#==========================================================

#Confusion Matrix: The first row is about the not-survived-predictions: 
    #493 passengers were correctly classified as not survived (called true negatives) 
    #and 56 where wrongly classified as not survived (false negatives).
#The second row is about the survived-predictions: 
    #93 passengers where wrongly classified as survived (false positives) 
    #and 249 where correctly classified as survived (true positives).
#A confusion matrix gives you a lot of information about how well your model does, 
#but theres a way to get even more, like computing the classifiers precision.
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(model, X_train, y_train, cv=5)
confusion_matrix_results = confusion_matrix(y_train, predictions)
print("\nConfusion Matrix: \n",confusion_matrix_results)
print("\nConfusion Matrix: \nThe first row is about the not-target-predictions:")
print("True negatives - correctly classified as not Target: ", confusion_matrix_results[0][0])
print("False negatives - wrongly classified as not Target: ",confusion_matrix_results[0][1])
print("\nThe second row is about the Target-predictions:")
print("False positives - wrongly classified as Target: ", confusion_matrix_results[1][0])
print("True positives - correctly classified as Target: " ,confusion_matrix_results[1][1]);pause()






#Precision and Recall: Our model predicts 81% of the time, a passengers survival correctly (precision). 
#The recall tells us that it predicted the survival of 73 % of the people who actually survived (Recall).
from sklearn.metrics import precision_score, recall_score
print("\nPrecision and Recall results:")
print("Precision - predicts % of the time, a Target correctly:", round(precision_score(y_train, predictions), 2))
print("Recall- tells us that it predicted the Target % of the correct:", round(recall_score(y_train, predictions),2))
  







#F-Score: You can combine precision and recall into one score, which is called the F-score. 
#The F-score is computed with the harmonic mean of precision and recall. 
#Note that it assigns much more weight to low values. As a result of that, 
#the classifier will only get a high F-score, if both recall and precision are high.
from sklearn.metrics import f1_score
print("\nF1-score - combine precision and recall into one score")
print(f1_score(y_train, predictions));pause()







# #Understanding model accuracy
# # from sklearn.metrics import accuracy_score - https://scikit-learn.org/stable/modules/model_evaluation.html
# y_pred = [0, 2, 1, 3]
# y_true = [0, 1, 2, 3]
# prediction_accuracy = accuracy_score(y_true, y_pred);
# print("\nprediction_accuracy:", round(prediction_accuracy,2))
# #normalizebool, default=True
# #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
# Num_correct_samples = accuracy_score(y_true, y_pred, normalize=False)
# print("Number of correct results: "+ str(Num_correct_samples) + " of total: " + str(len(y_pred)));pause()

# Models Accuracy
y_pred = model.predict(X_test);print("y_pred.shape: ",y_pred.shape)
y_test;print("y_true.shape: ",y_test.shape)
prediction_accuracy = accuracy_score(y_test, y_pred);
print("prediction_accuracy:", round(prediction_accuracy,2))
#normalize bool, default=True - #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
Num_correct_samples = accuracy_score(y_test, y_pred, normalize=False)
print("Number of correct results: "+ str(Num_correct_samples) + " of total: " + str(len(y_pred)));pause()





#get oob score
model.fit(X_train, y_train)
print("\nHypertuned - oob score:", round(model.oob_score_, 2)*100, "%")  ;pause()  






#Classification report
from sklearn.metrics import classification_report   
print("\nDetailed classification report for HyperTuned model:")
print("Train scores:")
y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred));pause()

print("Test scores:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred));pause()









#Precision Recall Curve: For each person the Random Forest algorithm has to classify, 
#it computes a probability based on a function 
    #and it classifies the person as survived (when the score is bigger the than threshold) 
    #or as not survived (when the score is smaller than the threshold). 
#That's why the threshold plays an important part.
from sklearn.metrics import precision_recall_curve
# getting the probabilities of our predictions
y_scores = model.predict_proba(X_train)
y_scores = y_scores[:,1]
precision, recall, threshold = precision_recall_curve(y_train, y_scores)
#function
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show();pause()


#ROC AUC Curve - Another way to evaluate and compare your binary classifier 
#is provided by the ROC AUC Curve. This curve plots the true positive rate (also called recall) 
#against the false positive rate (ratio of incorrectly classified negative instances), 
#instead of plotting the precision versus the recall.
from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)
plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show();pause()

#ROC AUC Score - The ROC AUC Score is the corresponding score to the ROC AUC Curve. 
#It is simply computed by measuring the area under the curve, which is called AUC.
#A classifiers that is 100% correct, would have a ROC AUC Score of 1 and 
#a completely random classiffier would have a score of 0.5.
from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("\nROC-AUC-Score - A classifiers that is 100% correct, \nwould have a ROC AUC Score of 1, score is :", r_a_score);pause()