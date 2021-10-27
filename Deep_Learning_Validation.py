# -*- coding: utf-8 -*-
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
                        if (P_ReCall > 0.1) and (N_ReCall > 0.1):
                            
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
                            if (P_ReCall1 > 0.1) and (N_ReCall1 > 0.1):
                                
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