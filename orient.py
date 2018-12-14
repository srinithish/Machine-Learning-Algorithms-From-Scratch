#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:28:28 2018

@author: nithish k

I have written adaboost, knn , forest in different modules and calling them here.
Please check,
knn.py, forest.py , adaboost.py for their respective code

Please refer the report in the pdf commited in the git hub


"""
from forest import forest
from adaboost import AdaBoost
from boostedforest import boostedForest
import knn
import pandas as pd
import numpy as np
import itertools as itr
import random
import collections as col
import math
import tqdm
import sys
import os
import pickle as pk

if len(sys.argv) < 5:
    print("Usage: \n./orient.py train train_file.txt model_file.txt [model]")
    sys.exit()

#train train_file.txt model_file.txt [model]
#nearest, adaboost, forest, or best.
(trainOrTest, train_test_file, model_file, model) = sys.argv[1:5]

#train train-data.txt forest_model.txt forest
#test test-data.txt forest_model.txt forest

if model == 'forest' :
    
    if trainOrTest == 'train':
        myForest = forest(maxDepth=5,numTrees= 20,verbose = False)
        TrainX,TrainY,TrainXID= myForest.getDataFromFile(train_test_file)
        myForest.trainForest(TrainX,TrainY)
        pk.dump(myForest,open(model_file,'wb'))
        
    if trainOrTest == 'test':
        try:
            myForest = pk.load(open(model_file,'rb'))
        except:
            print("output file has not been generated")
        
        if myForest.isTrained:
            Xtest,yTest,XtestID = myForest.getDataFromFile(train_test_file)
            finalPredictions = myForest.predict(Xtest)
            myForest.writeToFile(XtestID,finalPredictions,'output_forest.txt')
            myForest.writeToFile(XtestID,finalPredictions,'output.txt')
            print("Accuracy is: " ,100*sum(finalPredictions==yTest)/len(yTest))
        else:
            print("Untrained model being tested")
  
#train train-data.txt adaboost_model.txt adaboost
#test test-data.txt adaboost_model.txt adaboost    
            
elif model == 'adaboost' :
    
    if trainOrTest == 'train':
        myBoost = AdaBoost(200,verbose = False)
        TrainX,TrainY,TrainXID = myBoost.getDataFromFile(train_test_file)
        myBoost.train(TrainX,TrainY)
        pk.dump(myBoost,open(model_file,'wb'))
        
    if trainOrTest == 'test':
        try:
            myBoost = pk.load(open(model_file,'rb'))
        except:
            print("output file has not been generated")
        
        if myBoost.isTrained:
            Xtest,yTest,XtestID = myBoost.getDataFromFile(train_test_file)
            finalPredictions = myBoost.predict(Xtest)
            myBoost.writeToFile(XtestID,finalPredictions,'output_adaboost.txt')
            myBoost.writeToFile(XtestID,finalPredictions,'output.txt')
            print("Accuracy is: " ,100*sum(finalPredictions==yTest)/len(yTest))
        else:
            print("Untrained model being tested")

#train train-data.txt nearest_model.txt nearest
#test test-data.txt nearest_model.txt nearest
elif model == 'nearest' :
    
    if trainOrTest == 'train':
        knn.train(train_test_file,model_file)
        
    if trainOrTest == 'test':
        try:
            myKnn = open(model_file,'rb')
        except:
            print("output file has not been generated")
        
        finalPredictions,yTest,XtestID= knn.test(48,model_file ,train_test_file) 
        knn.writeToFile(XtestID,finalPredictions,'output_nearest.txt')
        knn.writeToFile(XtestID,finalPredictions,'output.txt')
        print("Accuracy is: ",knn.accuracy(finalPredictions,yTest))



#train train-data.txt best_model.txt best
#test test-data.txt best_model.txt best        
elif model == 'best' :
    
    if trainOrTest == 'train':
        knn.train(train_test_file,model_file)
        
    if trainOrTest == 'test':
        try:
            myKnn = open(model_file,'rb')
        except:
            print("output file has not been generated")
        
        finalPredictions,yTest,XtestID= knn.test(48,model_file ,train_test_file) 
        knn.writeToFile(XtestID,finalPredictions,'output_best.txt')
        knn.writeToFile(XtestID,finalPredictions,'output.txt')
        print("Accuracy is: " ,knn.accuracy(finalPredictions,yTest))    
    
else:
    print("Unknown model please check the model name ")
    
    
