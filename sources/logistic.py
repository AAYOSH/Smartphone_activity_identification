import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import preprocessing


def compute_macro_avg(y_true,y_predicted):
    
    '''
    Reference: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    Macro avg consiste em avaliar, para um conjunto de classes bem balanceados, i.e, se há 500 classes 1, valores devem
    ser proximos para outras classes também
    
    Para precisão, por exemplo:
    
    PR = SOMAR precisão de cada classe e dividir pelo numero de classes
    
    O mesmo para recall,
    
    F1_macro = 2*(Pmacro+Rmacro)/(Pmacro+Rmacro)
    
    Parametros de Entrada da função compute_macro_avg():
    
    Algorithm: algorithm class to evaluate
    
    y_true
    
    y_predicted
    
    
    
    Parametros de saída:
    
    Precision,recall,f1
    
   
    
    '''
    # cria matriz de confusao para as l classes
    matrix = confusion_matrix(y_true, y_predicted)
    
    # obtemos quantidade de classes a serem avaliadas
    l,_ = matrix.shape
    
    precision_v = []
    recall_v = []
    f1_v = []
    for class_ in range(l): # < matrix rows and columns starts at 0 index
        TP = matrix[class_,class_]
        FP = matrix[:,class_].sum() - TP
        FN = matrix[class_,:].sum() - TP
        
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)
        precision_v.append(precision)
        recall_v.append(recall)
        f1_v.append(f1)
    
    return sum(precision_v)/l,sum(recall_v)/l,sum(f1_v)/l


if __name__ == '__main__':


    X_test = pd.read_csv('../data/X_test.txt',delim_whitespace=True, header=None)
    X_train = pd.read_csv('../data/X_train.txt',delim_whitespace=True, header=None)
    y_test = pd.read_csv('../data/y_test.txt',delim_whitespace=True, header=None)
    y_train = pd.read_csv('../data/y_train.txt',delim_whitespace=True, header=None)

    #==================================================================================
    # baseline method for logistic regression 
    clf = LogisticRegression(multi_class = 'multinomial',solver = 'newton-cg').fit(X_train,y_train[0])
    y_predicted = clf.predict(X_test)

    _,_,f1  = compute_macro_avg(y_test,y_predicted)

    print(f"f1 metric : {f1}")