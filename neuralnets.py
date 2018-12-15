#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:01:18 2018

@author: nithish k
"""

import pandas as pd
import numpy as np
import itertools as itr
import random
import collections as col
import math
import tqdm
import time 
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

class networkLayer():
    ##dict of nodes
    
    def __init__(self,currentNumNodes,nextNumNodes):
        self._numNodes = currentNumNodes
        self.inputs = None
        self.outputs = None
        self.derivOutputs = None
        self.WeightsAtNodes = np.random.rand(currentNumNodes,nextNumNodes)
        self._activationFunc = None
        self.deltas = None
        
        pass
    
    def addNode(self,position, networkNode):
        self.Nodes[position] = networkNode
        pass
    
    def getWeights(self):
        return self.WeightsAtNodes
    def setWeights(self,weightMatrix):
        self.WeightsAtNodes = weightMatrix
        pass
    
    
    
    def setInputsToNodes(self,arrayOfInputs):
        self.inputs = arrayOfInputs
        
    def _sigmoidFunc(self,oneInput):
        output = 1/(1+math.exp(-oneInput))
        
        return output
    
    
    def _deriveSigmoidFunc(self,oneInput):
        output = self._sigmoidFunc(oneInput)*(1-self._sigmoidFunc(oneInput))
        return output
        
        
                
    def calcGetNodeOutputs(self,activationFuncName):
        self._activationFunc = activationFuncName
        
        if activationFuncName == 'None':
           self.outputs = self.inputs
           return self.outputs
       
        if activationFuncName == 'sigmoid':
            outputs = np.array(list(map(self._sigmoidFunc,self.inputs)))
            self.outputs = outputs
            return outputs
        
        if activationFuncName == 'relu':
            pass
        
    def getDerivatedOutputs(self):
        
        
        if self._activationFunc == 'None':
            derivOutputs =  np.ones(len(self.inputs))
            self.derivOutputs = derivOutputs 
            return derivOutputs
            
        if self._activationFunc == 'sigmoid':
            derivOutputs = np.array(list(map(self._deriveSigmoidFunc,self.inputs)))
            self.derivOutputs = derivOutputs
            return derivOutputs
        
        if self._activationFunc == 'relu':
            pass
    
    def getCachedOutputs(self):
        return self.outputs
    def setDeltas(self,deltas):
        self.deltas = deltas
        
    def getDeltas(self):
        return self.deltas
        
##weight matrix
            
      
        
class neuralNet():
    def __init__(self,numLayers,NodesPerLayer,
                 ContOrCatTarget,multiclass = False ,learningRate= 0.05 ,
                 epochs =30,earlyStoppingEpochs = 10, verbose = True):
        self._numLayers = numLayers ## not used yet
        self._NodesPerLayer = NodesPerLayer ##As list in sequence of Nodes [14,5,3] except the inputlayer
        self._listOfLayers = None
        self._learningRate = learningRate
        self._epochs =epochs
        self._ContOrCatTarget = ContOrCatTarget #Cont or Cat
        self._multiclass = multiclass
        self.earlyStoppingEpochs = earlyStoppingEpochs
        self.verbose = verbose
        pass
    
    
    def verbosePrint(self,*args):
        if self.verbose:
            print(*args) 
        

    
    def compileNetwork(self,XTrain):
        inputNumNodes = XTrain.shape[1]
        numOutputNodes = self._NodesPerLayer[-1]
        listOfLayers = []
        for numCurrentNodes,numNextNodes in zip([inputNumNodes]+self._NodesPerLayer,self._NodesPerLayer+[numOutputNodes]):
            listOfLayers.append(networkLayer(numCurrentNodes,numNextNodes))
            
        self._listOfLayers = listOfLayers #includes input and output layer layer
        
                
        
        
    def forwardPass(self,oneExampleAsArray):
        inputNodes = oneExampleAsArray
        lastLayerPosition = len(self._listOfLayers)-1
        
        for layerPosition,layer in enumerate(self._listOfLayers):
            if layerPosition == 0 :
                
                layer.setInputsToNodes(inputNodes)
                output = layer.calcGetNodeOutputs('None')
                weightTimesOutput = np.dot(output,layer.getWeights()) 
             
            
                    
                
            elif layerPosition < lastLayerPosition :
                layer.setInputsToNodes(weightTimesOutput)
                output = layer.calcGetNodeOutputs('sigmoid')
                weightTimesOutput = np.dot(output,layer.getWeights())
                
            elif layerPosition == lastLayerPosition:
                if self._ContOrCatTarget == 'Cont':
                    layer.setInputsToNodes(weightTimesOutput)
                    output = layer.calcGetNodeOutputs('None')
                    weightTimesOutput = np.dot(output,layer.getWeights())
                
                elif self._ContOrCatTarget == 'Cat':
                    layer.setInputsToNodes(weightTimesOutput)
                    output = layer.calcGetNodeOutputs('sigmoid')
                    weightTimesOutput = np.dot(output,layer.getWeights())
                    
        #last layer neglect weightTimesOutput
        return output #final output
        
    
    
    def backpropogate(self,yTrain):
        ##deltas are for each node
        
        
        for layerPosition,layer in enumerate(self._listOfLayers[::-1]):
            ##for the output layer
            if layerPosition == 0 :
                ##change loss function here
                
                deltas = layer.getDerivatedOutputs()*\
                            (yTrain-layer.getCachedOutputs())
                
                layer.setDeltas(deltas)
            
            
            ##for other than output layer
            else:
                
                
                deltas = layer.getDerivatedOutputs()*\
                         np.dot(layer.getWeights(),deltas)
                
                layer.setDeltas(deltas)
                
        ##update weights
        
        
    def updateWeights(self):
        
        ##updates weights to the respective layers
        for currentLayer,nextLayer in zip(self._listOfLayers,self._listOfLayers[1:]):
            
            currentWeights = currentLayer.getWeights()
           
            DeltaMatrix = np.tile(nextLayer.getDeltas(),(currentWeights.shape[0],1)) ##repeat delata so that matrix is formed
            ##alpha* ouput@i * delta @ nextlayer j
            outputTimesDeltaMat = self._learningRate*currentLayer.getCachedOutputs()[:,np.newaxis]*DeltaMatrix
            newWeights = currentWeights+outputTimesDeltaMat                
            currentLayer.setWeights(newWeights)
            
            
    
    def train(self,XTrain,YTrain):
        self.compileNetwork(XTrain) ##forms layers
        
        
        for epoch in range(self._epochs):
            
            
            for trainExampleX,trainExampleY in zip(XTrain,YTrain):
                self.forwardPass(trainExampleX) ##sets inputs
                self.backpropogate(trainExampleY) ##back propogates errors calculates deltas
                self.updateWeights() ##updates wieghts in place
            
            
            currentAccuracy = self.getAccuracy(XTrain,YTrain)
            self.verbosePrint("Epoch :" , epoch, "Accuracy :",currentAccuracy)
            
            
                
                
        pass
    
    def predict(self,XTest):
        if not self._multiclass:
            
            predictions = [self.forwardPass(testExamp)[0] for testExamp in XTest ]
        
        elif self._multiclass:
            
            predictions = [self.forwardPass(testExamp) for testExamp in XTest ]
        
        return np.array(predictions)
    
    def getAccuracy(self,X,y):
        predictions = self.predict(X)
        
        if self._ContOrCatTarget == 'Cont':
            return r2_score(y,predictions)
        
        if self._ContOrCatTarget == 'Cat':
            if not self._multiclass:
                finalPredictions = (predictions>=0.5).astype(int)
            elif self._multiclass:
                row_maxes = predictions.max(axis=1).reshape(-1, 1)
                finalPredictions = np.where(predictions == row_maxes, 1, 0)
                
            return accuracy_score(y,finalPredictions)
        
    
    def getDataFromFile(self, filename):
    
        DataDf = pd.read_csv(filename,header = None, sep = ' ' )
        DataDf.columns = ['photo_id','correct_orientation'] + [i-2 for i  in DataDf.columns[2:].tolist()]
    #        self.trainData = trainDataDf
        XDataMatrix = np.array(DataDf.loc[:,~DataDf.columns.isin(['photo_id','correct_orientation'])])
        YLabels = DataDf['correct_orientation']
        XDataID = DataDf['photo_id']
        return XDataMatrix,YLabels,XDataID
        
        
  
if __name__ == '__main__':
    
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    
#    ##cont
#    boston = load_boston()
#    
#    XTrian = boston['data']
#    yTrain = boston['target']
#    
    #cat
    from sklearn.datasets import load_breast_cancer
    breastCancer = load_breast_cancer()
    XTrian = breastCancer['data']
    yTrain = breastCancer['target']
    
    
    ##common
    XTrian = preprocessing.StandardScaler().fit_transform(XTrian)
#    numLayers,NodesPerLayer,
#                 ContOrCatTarget,learningRate= 0.05 ,epochs =30
    
    
    
    ##photo orientation
    

    
    X_train, X_test, y_train, y_test = train_test_split(XTrian, yTrain, test_size=0.1, random_state=42)
    
    
    myNet = neuralNet(3,[10,5,4],'Cat',True,0.02,50)
   
    
    
        ##photo orientation
    X_train,y_train,XDataID = myNet.getDataFromFile('train-data.txt')
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
#    
    y_train = pd.get_dummies(y_train)
    y_train = np.array(y_train)
    
    myNet.train(X_train,y_train)    
    
    X_test,y_test,XDataID = myNet.getDataFromFile('test-data.txt')
    X_test = preprocessing.StandardScaler().fit_transform(X_test)
    
    y_test = pd.get_dummies(y_test)
    y_test = np.array(y_test)
    
    
    Predictions = myNet.predict(X_test)
    
    print("test accu",myNet.getAccuracy(X_test,y_test))
    
    
    ## for cat
#    finalPredictions = (Predictions>=0.5).astype(int)

    
#    print("Accuracy is: " ,sum(finalPredictions==y_test)/len(y_test))
    
    ##for cont
    
    
    
#    print("accuracy is ",r2_score(y_test,Predictions))
    
    

            
        