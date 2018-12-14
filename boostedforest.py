#!/usr/bin/env python3
# -*- coding: utf-8 -*-




"""
Created on Sun Dec  2 11:19:11 2018

@author: nithish k

Please refer the report in the pdf commited in the git hub
Tried this for  the best model however gives a conssitent accuracy of 68%
"""
###Adabooseted forest

from forest import forest
from adaboost import AdaBoost
import pandas as pd
import numpy as np
import itertools as itr
import random
import collections as col
import math
import tqdm

class boostedForest(AdaBoost):
    
    def __init__(self,**kwargs):
        
        self.kwargs = kwargs
        AdaBoost.__init__(self,kwargs.get('nForests',5))
        self.verbose = kwargs.get('verbose',False)
        pass
        
    def verbosePrint(self,*args):
        if self.verbose:
            print(*args)
        
    def train(self,TrainXmatrix,TrainY):
        
        self.trainXMatrix = TrainXmatrix
        self.trainYLabels = TrainY
        
        uniqueYLables = set(TrainY)
        numYLabels = len(uniqueYLables)
        
        
        weightsForForests = col.defaultdict(int)
        numTrainObs = self.trainXMatrix.shape[0] 
        obsWeights = np.array([1/numTrainObs for i in range(numTrainObs)]) ##initialise
        
        ##create forests and train them
        trainedForests = [] ##multiple forests
        self.verbosePrint("\nNumber of forest building : ",self.nTrees )
        
        for i in range(self.nTrees): ##do not confuse with trees, they are forest
            ForestObj = forest(**self.kwargs)
            ForestObj.trainForest(TrainXmatrix,TrainY)
            trainedForests.append(ForestObj) #entire foreest
            self.verbosePrint("\nNumber of forests built : ", i+1)
            
        self.hypothesis = trainedForests ##assempble of forests
        
        
        self.verbosePrint("\nCalculating weights ....")
        
        for forestNum,ForestObj in enumerate(self.hypothesis): ##feature pair as tuple

            trainPredictionList = ForestObj.predict(TrainXmatrix)
               
            forestError  = self._calcError(obsWeights,trainPredictionList,self.trainYLabels)

      
            if forestError > 1-(1/numYLabels): #not better than random guessing
                continue
            
#            numSatisifyingStumps+=1
            forestWeight = math.log((1-forestError)/forestError) + math.log(numYLabels-1)
          
            obsWeights = self._adjustNormWeights(obsWeights,forestWeight,trainPredictionList,self.trainYLabels)
            
            
            weightsForForests[forestNum] =  forestWeight
        self._weightsForHypothesis = weightsForForests
        
    def predict(self,TestXmatrix):
        self.verbosePrint("\nPredicting.....")
        dictOfLabelsCumWeights = col.defaultdict(lambda: col.defaultdict(int))
        
        for forestNum,ForestObj in enumerate(self.hypothesis):

            predictionList = ForestObj.predict(TestXmatrix)
            
            decisionWeight = self._weightsForHypothesis[forestNum]
            
            for i,label in enumerate(predictionList):
                dictOfLabelsCumWeights[i][label] += decisionWeight
        
        dictOfLabelsCumWeights = dict(dictOfLabelsCumWeights)
        finalWeightedPredictions = \
        [max(dictOfLabelsCumWeights[i],key = dictOfLabelsCumWeights[i].get) for i in range(len(predictionList))] 
        return finalWeightedPredictions
    
    
    
    
if __name__ == '__main__':
    myBoostedForest = boostedForest(nForests = 10,numTrees = 10,verbose = True)
    TrainX,TrainY,TrainXID = myBoostedForest.getDataFromFile('train-data.txt')
   
    myBoostedForest.train(TrainX,TrainY)
    Xtest,yTest,XtestID  = myBoostedForest.getDataFromFile('test-data.txt')
    finalPredictions = myBoostedForest.predict(Xtest)
    myBoostedForest.writeToFile(XtestID,finalPredictions,'output.txt')
    
    print(sum(finalPredictions==yTest)/len(yTest))