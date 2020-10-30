# Implementing KNN classifier
import pandas as pd
import numpy as np
import math

# Datasets
train = pd.read_csv('spam_train.csv')
test  = pd.read_csv('spam_test.csv')
# K values
K = [1,5,11,21,41,61,81,101,201,401]

# Datasets
train = pd.read_csv('spam_train.csv')
test  = pd.read_csv('spam_test.csv')
# K values
K = [1,5,11,21,41,61,81,101,201,401]

attributes = [ 'f'+str(i) for i in range(1,58) ]
accuracy_List = []
# make training and testing set
train_set = train[attributes].values
test_set  = test[attributes].values

#1(a) without normalization
pred = {}
for i in range(test.shape[0]):
  sum_square = np.sum(np.square(test_set[i,:] - train_set), axis=1)
  df = pd.DataFrame({'sum_square':sum_square,'class':train['class']})
  sorted_class = df.sort_values('sum_square')['class']
  pred[i] = [sorted_class[:k].mean() > 0.5 for k in K]

test_pred = pd.concat([test, pd.DataFrame(pred, index=K).astype(int).T], axis=1)
# Percentage of accuracy
accuracy1 = pd.Series({k:(test_pred.Label == test_pred.loc[:,k]).mean() for k in K})
print(accuracy1)


#1(b) with z-score normalization

#before Normalize, set mu and sigma
mu = train_set.mean(axis=0)
sigma = train_set.std(axis=0)

#normalize data
z_train =  (train_set - mu)/sigma
z_test = (test_set - mu)/sigma

pred = {}
for i in range(test.shape[0]):

    sum_square = np.sum(np.square(z_test[i,:] - z_train), axis=1)
    df = pd.DataFrame({'sum_square':sum_square,'class':train['class']})

    sorted_class = df.sort_values('sum_square')['class']
    
    # Predict by determining whether mean of training class is > 5
    pred[i] = [sorted_class[:k].mean() > 0.5 for k in K]
    
test_pred_b= pd.concat([test, pd.DataFrame(pred, index=K).astype(int).T], axis=1)

# Percentage of accuracy ,  whether labels match each other
accuracy2 = pd.Series({k:(test_pred_b.Label == test_pred_b.loc[:,k]).mean() for k in K})
print(accuracy2)

#1(c)
c = test_pred_b.copy()
c[K] = c[K].applymap(lambda x:"no" if x==0 else 'spam')
print(c[[' ID']+K].head(50))