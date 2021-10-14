# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:31:59 2021

@author: micha
"""
# https://seaborn.pydata.org/tutorial/categorical.html
# Showing multiple relationships with facets



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)

#df = pd.read_csv("df_Data_for_saved_ML_data.csv",index_col=0)
filename = 'S1_train_Loan_Basic_Data_Cleaning.csv'

#Load df from file
# dataFrame from Hyptune used on the model was df shape was : (4694, 18) there we need 18 columns
df = pd.read_csv(filename)
print("\n<<Loaded df.shape: ", df.shape);print(df.info())#ratio 1(240):4(948)



##sns.catplot(x="day", y="total_bill", hue="smoker", col="time", aspect=.7,  kind="swarm", data=tipe)



sns.catplot(x="EXIST_MORTG_DEBT", y="EXIST_PROPERTY_VALUE", hue="BAD_LOAN", col="JOB", aspect=.7,  kind="swarm", data=df)