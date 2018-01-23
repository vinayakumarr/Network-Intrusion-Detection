#import pandas as pd
#import pandas_ml as pdml
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

predicted = np.genfromtxt ('res/predicted1.txt', delimiter=",")
expected = np.genfromtxt ('res/expected1.txt', delimiter=",")
y_true = expected[0:311029]
y_pred = predicted[0:311029]
#print(np.count_nonzero(y_true <> 2))
cm = confusion_matrix(y_true, y_pred)

y_true1 = np.array(y_true)
y_pred1 = np.array(y_pred)

print(accuracy_score(y_true1, y_pred1))
#print(cm)
target_names = ['class 0', 'class 1', 'class 2','class3','class4']
#print("**************Built-In****************************")
#print(classification_report(y_true, y_pred, target_names=target_names))
#print("*************************************************")
cl = 0;
for num in range(0,5):
	new_y_true = y_true.astype(int)
	new_y_pred = y_pred.astype(int)
	
	#Convert one class as 1 and rest of the classes as 0

	ipresent= np.where(new_y_true == cl)[0]
	new_y_true[ipresent] = 9
	inot = np.where(new_y_true <> 9)[0]
	new_y_true[inot] = 0
	new_y_true[ipresent] = 1
	ipresent= np.where(new_y_pred == cl)[0]
 	new_y_pred[ipresent] = 9
	inot = np.where(new_y_pred <> 9)[0]
	new_y_pred[inot] = 0
	new_y_pred[ipresent] = 1

	print ("---------------------------------------------------------\n"+ "CLASS " + str(cl) )
	#print( np.count_nonzero(new_y_true))
	#print( np.count_nonzero(new_y_pred))
	tn, fp, fn, tp = confusion_matrix(new_y_true, new_y_pred).ravel()
	
	#print(" TP:\t" + str(tp) + "\nTN:\t" + str(tn) + "\nFP:\t" + str(fp) + "\nFN:\t" + str(fn))
	# Measures are calculated according to https://en.wikipedia.org/wiki/Confusion_matrix,
	# If further measures are given in above page, can be extended easily.
	# Sensitivity, Recall, hit rate, or true positive rate (TPR) = TP/(TP+FN)
        tp = tp.astype(float)
	tn = tn.astype(float)
	fp = fp.astype(float)
	tn = tn.astype(float)
        
        # tpr
	tpr = tp/(tp+fn)
	# Precision = TP/(TP+FP)
	#prec = tp/(tp+fp)
	# False Positive Rate (FPR) = FP/(FP+TN)
	fpr = fp/(fp+tn)
	# Accuracy = (TP+TN)/(TP+FP+TN+FN)
	acc = (tp+tn)/(tp+fp+tn+fn)
        # recall 
        #rec = tp/(tp+fn)
	print("Accuracy = \t" + str(acc) + "\nFPR= \t "+ str(fpr) + "\nTPR= \t" + str(tpr))
	cl = cl + 1


	 


