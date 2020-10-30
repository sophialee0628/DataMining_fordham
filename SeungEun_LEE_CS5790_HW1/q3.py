#libraries
import pandas as pd
import numpy as np
from datasets import *
from functions import *

Folds=10
MSE_sum=0

# 1) what is the best choice of  value and the corresponding test set MSE for each of the six dataset?
df_train_x_list=[xM_train1,xM_train2,xM_train3,xM_train4,xM_train5,xM_train6]
df_train_y_list=[yM_train1,yM_train2,yM_train3,yM_train4,yM_train5,yM_train6]
MSE_test_list=list()

for j in range(6):
    x_train= df_train_x_list[j]
    y_train= df_train_y_list[j]

    fSize= int(len(y_train)/Folds)
    for i in range(Folds):
        x_test_fold = x_train[ i*fSize : (i+1)*fSize]
        y_test_fold = y_train[ i*fSize : (i+1)*fSize]
        
        x_train_fold = np.concatenate((x_train[ : i*fSize], x_train[ (i+1)*fSize : ]), axis=0)
        y_train_fold = np.concatenate((y_train[ : i*fSize], y_train[ (i+1)*fSize : ]), axis=0)
        
        w = wSol(x_train_fold, y_train_fold, 0, 150)
        
        MSE_sum += mse(x_test_fold,w, y_test_fold)
    MSE_test_list.append(MSE_sum/Folds)


print("3(a) Answer")
i=1
for mse in MSE_test_list:
    lambda_min = mse.argmin()
    MSE_min_cv = mse[lambda_min]
    print("for dataset",i,": lambda =", lambda_min, ",the least MSE =", MSE_min_cv)
    i+=1