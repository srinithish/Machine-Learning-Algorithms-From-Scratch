#!/usr/bin/env python3


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 01:40:00 2018

@author: nithish k
Please refer the report in the pdf commited in the git hub
"""


import pandas as pd
import numpy as np
import itertools as itr
import random
import collections as col
import math
import time

class AdaBoost():
    
    
    def __init__(self,nTrees = 50 ,**params):
        
        self.trainData = []
        self.nTrees = nTrees
        self.verbose = params.get('verbose',False)
        self.isTrained = False
        self._decisionsStumps = None ##feature pair as key
        self._stumpWeights = None
        pass
    
    def verbosePrint(self,*args):
        if self.verbose:
            print(*args)
            
    def _getDecisionPairs(self,numStumps):
        
        random.seed(42)
        combinations = itr.combinations(range(self.trainXMatrix.shape[1]),2)
#        randomCombinations = random.choices(list(combinations),k=numStumps)
        
        
        return list(combinations)
    
    def _makeDecisions(self,featurePair):
        ##rerutns segregated observations
        
        majorityClass = col.defaultdict(int)
        feature1 , feature2 = featurePair ##tuple (x1,x2)
        
        booleanList = self.trainXMatrix[:,feature1] >= self.trainXMatrix[:,feature2]
        
        ##mejority class in positive decision
        majorityClass['Positive'] = col.Counter(self.trainYLabels[booleanList]).most_common(1)[0][0]
        
        negatedBooleanList = np.logical_not(booleanList)
        
        ##mejority class in negative decision
        majorityClass['Negative'] = col.Counter(self.trainYLabels[negatedBooleanList]).most_common(1)[0][0]
        
        mappingDict = {True: majorityClass['Positive'],False: majorityClass['Negative']}
        
        
        predictionList = pd.Series(booleanList).map(mappingDict)
        
        
        
        return majorityClass,predictionList
    
    def _calcError(self,obsWeights,predictionList, YActual):
        ##pass obsWeights as np.array
        
        misClassifiedList = predictionList != YActual
        weightedError = sum(np.array(obsWeights)[misClassifiedList])/sum(obsWeights)
        
#        print(weightedError)
        return weightedError
    
    def _adjustNormWeights(self,currentObsWeights,stumpWeight,predictionList,YActual):
        
        updatedObsWeights = currentObsWeights.copy()
        misClassifiedList = predictionList != YActual
        updatedObsWeights[misClassifiedList] = updatedObsWeights[misClassifiedList]*math.exp(stumpWeight)
        sumObsWeights = np.sum(updatedObsWeights)
        updatedObsWeights = updatedObsWeights/sumObsWeights
        
        return updatedObsWeights
    
    
    
    def train(self,TrainXmatrix,TrainY):
        
        
        self.trainXMatrix = TrainXmatrix
        self.trainYLabels = TrainY
        uniqueYLables = set(TrainY)
        numYLabels = len(uniqueYLables)
        AllfeatureCombinations = self._getDecisionPairs(self.nTrees)

        decisionsForFeatures = col.defaultdict(dict)
        weightsForDecisions = col.defaultdict(int)
        numTrainObs = self.trainXMatrix.shape[0] 
        obsWeights = np.array([1/numTrainObs for i in range(numTrainObs)]) ##initialise
        numSatisifyingStumps =0
        
        
        while numSatisifyingStumps < self.nTrees:
            
            
            randomCombinations = random.choices(AllfeatureCombinations,k=self.nTrees)
            AllfeatureCombinations = list(set(AllfeatureCombinations)-set(randomCombinations))
            
            if len(AllfeatureCombinations)==0:
                break
            for featurePair in randomCombinations: ##feature pair as tuple
    
                decisionStump, trainPredictionList = self._makeDecisions(featurePair)
                   
                stumpError  = self._calcError(obsWeights,trainPredictionList,self.trainYLabels)
    
   
               
                if stumpError > 1-(1/numYLabels): #not better than random guessing
                    continue
                
                numSatisifyingStumps+=1
                stumpWeight = math.log((1-stumpError)/stumpError) + math.log(numYLabels-1)
              
                obsWeights = self._adjustNormWeights(obsWeights,stumpWeight,trainPredictionList,self.trainYLabels)
                
                decisionsForFeatures[featurePair] = decisionStump
                weightsForDecisions[featurePair] =  stumpWeight
            self.verbosePrint("Gathered useful stumps :" ,numSatisifyingStumps)
            
        self._decisionsStumps = decisionsForFeatures ##feature pair as key
        self._stumpWeights = weightsForDecisions ##
        self.isTrained = True

    
    def getDataFromFile(self, filename):
    
        DataDf = pd.read_csv(filename,header = None, sep = ' ' )
        DataDf.columns = ['photo_id','correct_orientation'] + [i-2 for i  in DataDf.columns[2:].tolist()]
   
        XDataMatrix = np.array(DataDf.loc[:,~DataDf.columns.isin(['photo_id','correct_orientation'])])
        YLabels = DataDf['correct_orientation']
        XDataID = DataDf['photo_id']
        return XDataMatrix,YLabels,XDataID
    
    
    def predict(self, TestXmatrix):
        
        dictOfLabelsCumWeights = col.defaultdict(lambda: col.defaultdict(int))
        self.verbosePrint("\nPredicting.....")
        for (feature1,feature2),decisionNode, in self._decisionsStumps.items():
            booleanList = TestXmatrix[:,feature1] >= TestXmatrix[:,feature2]
            
            mappingDict = {True: decisionNode['Positive'],False: decisionNode['Negative']}
        
        
            predictionList = pd.Series(booleanList).map(mappingDict)
            
            decisionWeight = self._stumpWeights[(feature1,feature2)]
            
    
            for i,label in enumerate(predictionList):
                dictOfLabelsCumWeights[i][label] += decisionWeight
        
        dictOfLabelsCumWeights = dict(dictOfLabelsCumWeights)
        finalWeightedPredictions = \
        [max(dictOfLabelsCumWeights[i],key = dictOfLabelsCumWeights[i].get) for i in range(len(predictionList))] 
        
        return finalWeightedPredictions
    
    def writeToFile(self,ID,predictionList,filename):
        OutputDf = pd.DataFrame({'ID':ID,'Predictions':predictionList})
        OutputDf.to_csv(path_or_buf = filename ,sep = ' ',header = False ,index = False)
    
    
if __name__ == '__main__':
    myBoost = AdaBoost(200,verbose = True)
    TrainX,TrainY,TrainXID = myBoost.getDataFromFile('train-data.txt')
    #print(myBoost.train('Data'))
    start = time.time()
    myBoost.train(TrainX,TrainY)
    Xtest,yTest,XtestID = myBoost.getDataFromFile('test-data.txt')
    
    finalPredictions = myBoost.predict(Xtest)
    
    myBoost.writeToFile(XtestID,finalPredictions,'output.txt')
    print("Time elapsed is ", time.time()-start)
    print("Accuracy is: " ,sum(finalPredictions==yTest)/len(yTest))

   



