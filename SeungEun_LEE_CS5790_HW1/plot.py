import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import values from the hw1_2
from datasets import *
from q2 import *

#Plotting for the problem 
plt.subplot(231)
MSEtrain1_plot = plt.plot(MSE_ds1, label='Train MSE', color = 'blue')
MSEtest1_plot = plt.plot(MSE_test1, label='Test MSE', color = 'red')
plt.title("Dataset 1")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()

plt.subplot(232)
MSEtrain2_plot = plt.plot(MSE_ds2, label='Train MSE', color = 'blue')
MSEtest2_plot = plt.plot(MSE_test2, label='Test MSE', color = 'red')
plt.title("Dataset 2")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()


plt.subplot(233)
MSEtrain3_plot = plt.plot(MSE_ds3, label='Train MSE', color = 'blue')
MSEtest3_plot = plt.plot(MSE_test3, label='Test MSE', color = 'red')
plt.title("Dataset 3")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()

plt.subplot(234)
MSEtrain4_plot = plt.plot(MSE_ds4, label='Train MSE', color = 'blue')
MSEtest4_plot = plt.plot(MSE_test4, label='Test MSE', color = 'red')
plt.title("Dataset 4")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()

plt.subplot(235)
MSEtrain5_plot = plt.plot(MSE_ds4, label='Train MSE', color = 'blue')
MSEtest5_plot = plt.plot(MSE_test4, label='Test MSE', color = 'red')
plt.title("Dataset 5")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()

plt.subplot(236)
MSEtrain6_plot = plt.plot(MSE_ds6, label='Train MSE', color = 'blue')
MSEtest6_plot = plt.plot(MSE_test6, label='Test MSE', color = 'red')
plt.title("Dataset 6")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.savefig('/u/erdos/csga/slee374/Desktop/datamining/HW1/all6.png')