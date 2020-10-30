
import pandas as pd
import numpy as np


"""All pre functions are below
    1) toMatrix: Making Matrix form >> using (to_numpy)
    2) wSol: weight closed form solution
    3) mse : calculating mean square errors
 """

def toMatrix(df,col):
    #for X
    X_tr= df.iloc[:,:col]
    XM_tr = X_tr.values
    #for Y
    Y_tr = df.iloc[:,col]
    YM_tr = Y_tr.values
    return(XM_tr, YM_tr)

def wSol(X_tr, Y_tr, lambda_start, lambda_last):
    # varying lambda values
    lambda_val = np.arange(lambda_start,lambda_last + 1)
    # List to store weights with varying lambdas
    wL =[]
    for Lambda in lambda_val:
        # w = (XtX + LI)-1 Xty
        # first part: XTX+LI
        Xt= np.transpose(X_tr) 
        XtX =np.dot(Xt,X_tr)
        I = np.identity(len(XtX))
        firstPart = XtX + np.dot(Lambda, I)
        # inverse of first part
        firstPart_inv = np.linalg.inv(firstPart)
        # second part
        XtY = np.dot(Xt, Y_tr)

        # Weight solution!
        ws = np.dot(firstPart_inv, XtY)
        # save weights to the array
        wL.append(ws.flatten())
        w_train_ds = np.transpose(np.array(wL))

    return(w_train_ds)

def mse(X_tr, w, Y_tr):
    Y_p = np.dot(X_tr, w)
    # Set mean squared error to zero
    sum_error = 0.0
    for i in range(len(Y_tr)):
        Y_pred_error = Y_p[i] - Y_tr[i]
        sum_error += (Y_pred_error ** 2)
    MSE = sum_error / float(len(Y_tr)) 
    return MSE