#2020/09/11 SEUNG EUN LEE

#This python file is data set files.
# you can use dataset by using "import" from other python files

#libraries
import pandas as pd

import numpy as np

#import functions from function.py
import functions as fc

#first, read the file
ds_1000=pd.read_csv('train-1000-100.csv')

# creating additional training files
train_50=ds_1000.iloc[:50]
train_100=ds_1000.iloc[:100]
train_150=ds_1000.iloc[:150]

# changing it to csv files
train_50.to_csv('train-50(1000)-100.csv')
train_100.to_csv('train-100(1000)-100.csv')
train_150.to_csv('train-150(1000)-100.csv')

# datasets
train1 = pd.read_csv('train-100-10.csv')
train2 = pd.read_csv('train-100-100.csv')
train3 = pd.read_csv('train-1000-100.csv')
train4 = pd.read_csv('train-50(1000)-100.csv', index_col = 0)
train5 = pd.read_csv('train-100(1000)-100.csv', index_col = 0)
train6 = pd.read_csv('train-150(1000)-100.csv', index_col = 0)

# test data sets
test1 = pd.read_csv('test-100-10.csv')
test2 = pd.read_csv('test-100-100.csv')
test3 = pd.read_csv('test-1000-100.csv')

# Make all Datasets to Matrix

xM_train1, yM_train1 = fc.toMatrix(train1,10)
xM_train2, yM_train2 = fc.toMatrix(train2,100)
xM_train3, yM_train3 = fc.toMatrix(train3,100)
xM_train4, yM_train4 = fc.toMatrix(train4,100)
xM_train5, yM_train5 = fc.toMatrix(train5,100)
xM_train6, yM_train6 = fc.toMatrix(train6,100)
xM_test1, yM_test1 = fc.toMatrix(test1,10)
xM_test2, yM_test2 = fc.toMatrix(test2,100)
xM_test3, yM_test3 = fc.toMatrix(test3,100)

