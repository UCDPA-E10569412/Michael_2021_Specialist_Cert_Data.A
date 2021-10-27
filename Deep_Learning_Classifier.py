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

# scatter_plot_Recall_Pos_True_(loop_number, Model_performance['P_ReCall'], Model_performance['N_ReCall'], Model_performance['Optimiser'])
 