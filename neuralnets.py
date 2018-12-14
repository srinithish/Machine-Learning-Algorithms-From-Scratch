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


class networkNode():
    
    def __init__(self,inputVal = None):
        self.inputVal = inputVal
        self.outputVal = None
        
        pass
    
    def setInputVal(self, value):
        self.inputVal = value
        pass
    
    def getOutputVal(self,stepFuncName = 'sigmoid'):
        if stepFuncName == 'sigmoid':
            output = 1/(1+math.exp(-self.inputVal))
            self.outputVal = output
            return output


    
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
            self.derivOutputs =derivOutputs 
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
    def __init__(self,numLayers,NodesPerLayer,learningRate,epochs):
        self._numLayers = numLayers ## including input and output
        self._NodesPerLayer = NodesPerLayer ##As list in sequence of Nodes [14,5,3]
        self._listOfLayers = None
        self._learningRate = learningRate
        pass
    
    def getDataFromFile(self, filename): ##need to change for other data
    
        DataDf = pd.read_csv(filename,header = None, sep = ' ' )
        DataDf.columns = ['photo_id','correct_orientation'] + [i-2 for i  in DataDf.columns[2:].tolist()]
    #        self.trainData = trainDataDf
        XDataMatrix = np.array(DataDf.loc[:,~DataDf.columns.isin(['photo_id','correct_orientation'])])
        YLabels = DataDf['correct_orientation']
        XDataID = DataDf['photo_id']
        return XDataMatrix,YLabels,XDataID
    
    
    def train(XTrain):
        
        
        pass
    
    def compileNetwork(self,XTrain):
        inputNumNodes = XTrain.shape[1]
        numOutputNodes = self._NodesPerLayer[-1]
        listOfLayers = []
        for numCurrentNodes,numNextNodes in zip([inputNumNodes]+self._NodesPerLayer,self._NodesPerLayer+[numOutputNodes]):
            listOfLayers.append(networkLayer(numCurrentNodes,numNextNodes))
            
        self._listOfLayers = listOfLayers #includes input and output layer layer
        
                
        
        
    def forwardPass(self,oneExampleAsArray):
        inputNodes = oneExampleAsArray
        
        
        for layerPosition,layer in enumerate(self._listOfLayers):
            if layerPosition == 0 :
                
                layer.setInputsToNodes(inputNodes)
                output = layer.calcGetNodeOutputs('None')
                weightTimesOutput = np.dot(output,layer.getWeights()) 
                
            else :
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
        
        for currentLayer,nextLayer in zip(self._listOfLayers,self._listOfLayers[1:]):
            
            currentWeights = currentLayer.getWeights()
            
            DeltaMatrix = np.tile(nextLayer.getDeltas(),(currentWeights.shape[0],1)) ##repeat delata so that matrix is formed
            outputTimesDeltaMat = currentLayer.getCachedOutputs()[:,np.newaxis]*DeltaMatrix
            newWeights = currentWeights+outputTimesDeltaMat                
            currentLayer.setWeights(newWeights)
            
            
    
        
        
        
        pass
    
    
XTrian = np.array([[1,2,3],[2,3,4]])
yTrain = np.array([1,2,3])
myNet = neuralNet(1,[3,3],0.05,10)
myNet.compileNetwork(XTrian) 
    
myNet.forwardPass(XTrian[0,:])        
myNet.backpropogate(yTrain)
myNet.updateWeights()
myNet._listOfLayers[1].getWeights()
        
        
        