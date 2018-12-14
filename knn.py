#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:51:06 2018

@author: 18123
"""

"""
KNN - 

At the outset, we have loaded the training data to the to our respective files,
Please refer the report in the pdf commited in the git hub 



"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt



t1 = time.time()

def loadData(filename):

    DataDf = pd.read_csv(filename,header = None, sep = ' ' )
    DataDf.columns = ['photo_id','correct_orientation'] + [i-2 for i  in DataDf.columns[2:].tolist()]
   
    XDataMatrix = np.array(DataDf.loc[:,~DataDf.columns.isin(['photo_id','correct_orientation'])])
    YLabels = DataDf['correct_orientation']
    XDataID = DataDf['photo_id']
    return XDataMatrix,YLabels,XDataID

"""
theoptimisation of calc_dist function is inspired by thoughts at
https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
"""


def calc_dist(test_matrix,train_matrix):
    dists = np.sum(train_matrix**2,axis=1) + \
    np.sum(test_matrix**2, axis=1)[:, np.newaxis]+ \
    (-2 * np.dot(test_matrix, train_matrix.T))  
    return dists
    






#Sorting the distances and storing the labels 
def sortLabelsbyDist(distance_rotlabels_matrix,testX):
    labels = []
    for i in range(0,testX.shape[0],1):
        dt = np.array((distance_rotlabels_matrix[:,i],distance_rotlabels_matrix[:,testX.shape[0]]))
        dt = dt.T
        dt_sort = dt[dt[:,0].argsort()]
        labels.append(dt_sort[:,1])
    return labels





#Accessing K neighbours
def kneigh(k,labels,testX):
    predictions = []
    for i in range(0,testX.shape[0],1):
        p = np.bincount(labels[i][:k]).argmax()
        predictions.append(p)
    return np.array(predictions)


def accuracy(pred,true):
    
    acc = sum(pred==true)/len(true)
    
#    print("KNN Accuracy:",acc)
    return 100 * acc

def plot_knn():
    k_val = [i for i in range(40,50,2)]
    score = []
    for val in k_val:
        y_pred = kneigh(val)
        ac = accuracy(y_pred,y_true)
        score.append(ac)
    plt.plot(k_val,score)
    plt.xlabel("Values of K")
    plt.ylabel("Accuracies")    
    Best_k_Value = k_val[score.index(max(score))]  
    Best_score = max(score)
    print("Best K value is:", Best_k_Value)
    print("Best Score is:",Best_score )
    


def train(filename,model_file):
    
    
    DataDf = pd.read_csv(filename,header = None, sep = ' ' )
    DataDf.to_csv(model_file,header = None, sep = ' ' ,index = False)
    
    
    pass



def test(K,trainedModelFile,testfile):
    
    
    trainX,train_Y,trainXID = loadData(trainedModelFile)
    testX,test_Y , testXID = loadData(testfile)
    distance_matrix = calc_dist(testX,trainX).T
    distance_rotlabels_matrix = np.column_stack((distance_matrix, train_Y))
    labels = sortLabelsbyDist(distance_rotlabels_matrix,testX)
    yPred = kneigh(K,labels,testX)
    
    return yPred,test_Y,testXID

def writeToFile(ID,predictionList,filename):
    OutputDf = pd.DataFrame({'ID':ID,'Predictions':predictionList})
    OutputDf.to_csv(path_or_buf = filename ,sep = ' ',header = False ,index = False)


if __name__ == "__main__":
    
    
    train("train-data.txt","knn_model.txt")
    yPred,test_Y,testXID= test(48, "knn_model.txt","test-data.txt")
    testXID[yPred!=test_Y]
    print(accuracy(yPred,test_Y))
#    trainX,train_Y,trainXID = loadData("train-data.txt")
#    
#    
#    
#    testX,test_Y , testXID = loadData("test-data.txt")
#
#    distance_matrix = calc_dist(testX,trainX).T
#    distance_rotlabels_matrix = np.column_stack((distance_matrix, train_Y))
#    labels = sortLabelsbyDist(distance_rotlabels_matrix,testX)
#    yPred = kneigh(42,labels)
#    print(accuracy(yPred,test_Y))

    
    
    
    