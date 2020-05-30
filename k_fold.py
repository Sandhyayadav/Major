from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn import metrics
import Genetic_Algo as ga
__author__ = 'sandhya'
from sklearn.model_selection import cross_val_score,KFold
import ANN as NN
cnf=np.zeros(2)

kf=sklearn.model_selection.KFold(n_splits=10,shuffle=False,random_state=None)
for train,test in kf.split(NN.x):

    
    y_test=NN.y.loc[test]
    y_pred=ga.genetic(NN.x,NN.y,NN.x.loc[test])
    confusion = confusion_matrix(y_test,y_pred)
    cnf=cnf+confusion
    accuracy=accuracy_score(y_test, y_pred)


print ("Confusion matrix : ",cnf)
print("Accuracy is :::: ",accuracy)
