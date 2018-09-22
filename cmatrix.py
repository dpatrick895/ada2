import numpy as np 
import pandas as pd

def cmatrix(test_class_var, predictions):

	cmatrix = pd.crosstab(test_class_var, predictions, rownames=['Actual'], colnames=['Predicted'])

# Sometimes the model only predicts 1's, so we have calculations for each case
	if cmatrix.shape[1] == 2:
	    TN,TP,FN,FP = cmatrix.iloc[0, 0],cmatrix.iloc[1, 1],cmatrix.iloc[1, 0],cmatrix.iloc[0, 1]
	    errors = FN + FP
	    correct = TP + TN
	    total = TN + TP + FN + FP
	    print ('accuracy =',correct/total,': How many did we get correct?')
	    print ('precision =',TP/(TP+FP),': When we predict an increase, how often are we correct?')
	    print ('recall =',TP/(TP+FN),': How many of the increases did we "detect"?')
	else:
	    FP,TP = cmatrix.iloc[0, 0],cmatrix.iloc[1, 0]
	    errors = FP
	    correct = TP
	    total = TP + FP
	    print ('The model only predicted 1s')
	    print ('accuracy/precision =',correct/total)
	return cmatrix