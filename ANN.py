__author__ = 'Sandhya'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

fields=['DEFECT']
datacol=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fields)

datacol['DEFECT']=datacol['DEFECT'].replace(['yes','no'],[1,0])
y=datacol;
fieldsrow=['WMC','DIT','NOC','CBO','RFC','LCOM','Ca','Ce','NPM','LCOM3','LOC','DAM','MOA','MFA','CAM','IC','CBM','AMC']
datarow=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fieldsrow)
x=datarow;
x['WMC']=x['WMC'].apply(lambda x:((x-1)/(213-1)))
x['DIT']=x['DIT'].apply(lambda x:((x-0)/(4-0)))
x['NOC']=x['NOC'].apply(lambda x:((x-0)/(4-0)))
x['CBO']=x['CBO'].apply(lambda x:((x-0)/(20-0)))
x['RFC']=x['RFC'].apply(lambda x:((x-2)/(214-2)))
x['LCOM']=x['LCOM'].apply(lambda x:((x-0)/(22578-0)))
x['Ca']=x['Ca'].apply(lambda x:((x-0)/(16-0)))
x['Ce']=x['Ce'].apply(lambda x:((x-0)/(17-0)))
x['NPM']=x['NPM'].apply(lambda x:((x-0)/(212-0)))
x['LCOM3']=x['LCOM3'].apply(lambda x:((x-1)/(2-1)))
x['LOC']=x['LOC'].apply(lambda x:((x-6)/(1100-6)))
x['DAM']=x['DAM'].apply(lambda x:((x-0)/(1-0)))
x['MOA']=x['MOA'].apply(lambda x:((x-0)/(37-0)))
x['AMC']=x['AMC'].apply(lambda x:((x-0)/(5-0)))
#print(x)


syn0 = 2*np.random.random((18,1)) - 1 #randomize intial weights (Theta)

def runForward(X, theta): #this runs our net and returns the output
   return sigmoid(np.dot(X, theta))
def costFunction(X, y, theta): #our cost function, simply determines the arithmetic difference between the expected y and our actual y
   m = float(len(X))
   hThetaX = np.array(runForward(X, theta))
   return np.sum(np.abs(y - hThetaX))
def sigmoid(x): return 1 / (1 + np.exp(- x)) #Just our run-of-the-mill sigmoid function

