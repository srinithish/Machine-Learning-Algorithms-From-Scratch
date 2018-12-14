# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:20:20 2018

@author: nithish k
"""

    XTrian = np.array([[1,2,3],[2,3,4]])
    yTrain = np.array([1,2,3])
    myNet = neuralNet(3,[3,3],0.05,10)
    myNet.compileNetwork(XTrian) 
        
    myNet.forwardPass(XTrian[0,:])        
    myNet.backpropogate(yTrain)
    myNet.updateWeights()
    myNet._listOfLayers[1].getWeights()
    
        ##trails
#    myNet = neuralNet(3,[3,1],0.05,10)
#    myNet.compileNetwork(XTrian) 
#        
#    myNet.forwardPass(XTrian[0,:])        
#    myNet.backpropogate(yTrain[0])
#    myNet._listOfLayers[1].getWeights()
#    myNet._listOfLayers[1].getCachedOutputs()
#    myNet._listOfLayers[1].getDeltas()
#    myNet.updateWeights()
#    myNet._listOfLayers[1].getWeights()