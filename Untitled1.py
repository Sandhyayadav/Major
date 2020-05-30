#!/usr/bin/env python
# coding: utf-8

# In[3]:


__author__ = 'Shubham'
import random, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import math
import Untitled as NN

params = [100, 0.05, 100, 18,
          20]  # [Init pop (pop=100), mut rate (=5%), num generations , chromosome/solution length (18), # winners/per gen]
def genetic(a,b,c) :
 curPop = np.random.choice(np.arange(-0.2367, 0.2367, step=0.0001), size=(params[0], params[3]),replace=False)  # initialize current population to random values within range
 nextPop = np.zeros((curPop.shape[0], curPop.shape[1]))
 fitVec = np.zeros((params[0], 2))  # 1st col is indices, 2nd col is cost
 for i in range(params[2]):  # iterate through num generations
    fitVec = np.array([np.array([x, np.sum(NN.costFunction(a,b, curPop[x].reshape(18, 1)))]) for x in range(params[0])])  #Create vec of all errors from cost function
    
    winners = np.zeros((params[4], params[3]))  
    for n in range(len(winners)):  
        selected = np.random.choice(range(len(fitVec)), params[4], replace=False)
        wnr = np.argmin(fitVec[selected, 1])
        winners[n] = curPop[int(fitVec[selected[wnr]][0])]
    nextPop[:len(winners)] = winners  #populate new gen with winners
    nextPop[len(winners):] = np.array([np.array(np.random.permutation(np.repeat(winners[:, x], ((params[0] - len(winners)) / len(winners)), axis=0)))for x in range(winners.shape[1])]).T  #Populate the rest of the generation with offspring of mating pairs
    nextPop = np.multiply(nextPop, np.matrix([np.float(np.random.normal(0,2,1)) if random.random() < params[1] else 1 for x in range(nextPop.size)]).reshape(nextPop.shape))
    curPop = nextPop
 best_soln = curPop[np.argmin(fitVec[:, 1])]
 return  np.round(NN.runForward(c,best_soln.reshape(18, 1)));

