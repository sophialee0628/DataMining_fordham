#libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import *

# 1) which value gives the least test set MSE? lambda 0 to 150
wDs_train1 = fc.wSol(xM_train1, yM_train1,0, 150)
MSE_ds1= fc.mse(xM_train1,wDs_train1, yM_train1)
MSE_test1= fc.mse(xM_test1,wDs_train1, yM_test1)

wDs_train2 = fc.wSol(xM_train2, yM_train2,0, 150)
MSE_ds2= fc.mse(xM_train2,wDs_train2, yM_train2)
MSE_test2= fc.mse(xM_test2,wDs_train2, yM_test2)

wDs_train3 = fc.wSol(xM_train3, yM_train3,0, 150)
MSE_ds3= fc.mse(xM_train3,wDs_train3, yM_train3)
MSE_test3= fc.mse(xM_test3,wDs_train3, yM_test3)

wDs_train4 = fc.wSol(xM_train4, yM_train4,0, 150)
MSE_ds4= fc.mse(xM_train4,wDs_train4, yM_train4)
MSE_test4= fc.mse(xM_test3,wDs_train4, yM_test3)

wDs_train5 = fc.wSol(xM_train5, yM_train5,0, 150)
MSE_ds5= fc.mse(xM_train5,wDs_train5, yM_train5)
MSE_test5= fc.mse(xM_test3,wDs_train5, yM_test3)

wDs_train6 = fc.wSol(xM_train6, yM_train6,0, 150)
MSE_ds6= fc.mse(xM_train6,wDs_train6, yM_train6)
MSE_test6= fc.mse(xM_test3,wDs_train6, yM_test3)


MSE_values = [MSE_test1, MSE_test2, MSE_test3, MSE_test4, MSE_test5, MSE_test6]

# 2) lambda 1 to 150
wDs_train2_2 = fc.wSol(xM_train2, yM_train2,1, 150)
MSE_ds2_2= fc.mse(xM_train2,wDs_train2_2, yM_train2)
MSE_test2_2= fc.mse(xM_test2,wDs_train2_2, yM_test2)

wDs_train4_2 = fc.wSol(xM_train4, yM_train4,1, 150)
MSE_ds4_2= fc.mse(xM_train4,wDs_train4_2, yM_train4)
MSE_test4_2= fc.mse(xM_test3,wDs_train4_2, yM_test3)

wDs_train5_2 = fc.wSol(xM_train5, yM_train5,1, 150)
MSE_ds5_2= fc.mse(xM_train5,wDs_train5_2, yM_train5)
MSE_test5_2= fc.mse(xM_test3,wDs_train5_2, yM_test3)


# Anwser part: 
print("2(a) Answer")
i=1
for mse in MSE_values:
    lambda_min = mse.argmin()
    MSE_min = mse[lambda_min]
    print("for dataset",i,": lambda =", lambda_min, ",the least MSE =", MSE_min)
    i+=1

# 2(b)
plt.subplot(131)
MSEtrain2_2 = plt.plot(MSE_ds2_2, label='Train MSE', color = 'blue')
MSEtest2_2 = plt.plot(MSE_test2_2, label='Test MSE', color = 'red')
plt.title("Dataset:100-100.csv ")
plt.xlabel('lambda')
plt.ylabel('Mean Square Error')
plt.legend(loc='lower right')

plt.subplot(132)
MSEtrain4_2 = plt.plot(MSE_ds4_2, label='Train MSE', color = 'blue')
MSEtest4_2 = plt.plot(MSE_test4_2, label='Test MSE', color = 'red')
plt.title("Dataset:50(1000)-100")
plt.xlabel('lambda')
plt.ylabel('Mean Square Error')
plt.legend(loc='lower right')

plt.subplot(133)
MSEtrain5_2 = plt.plot(MSE_ds5_2, label='Train MSE', color = 'blue')
MSEtest5_2 = plt.plot(MSE_test5_2, label='Test MSE', color = 'red')
plt.title("Dataset:100(1000)-100")
plt.xlabel('lambda')
plt.ylabel('Mean Square Error')
plt.legend(loc='lower right')
plt.show()

plt.savefig('/u/erdos/csga/slee374/Desktop/datamining/HW1/2b.png')