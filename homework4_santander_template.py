import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# randomly arrange training images and labels
rand_order = np.random.permutation(len(y))
X = X[rand_order]
y = y[rand_order]
# Split into train/test folds
mid = int(len(X)/2)
X_train = X[0:200]
X_test = X[200:400]
y_train = y[0:200]
y_test = y[200:400]

# Linear SVM
lin_svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
lin_svm.fit(X_train, y_train)

# Non-linear SVM (polynomial kernel)
# TODO

# Apply the SVMs to the test set
yhat1 = lin_svm.decision_function(X_test)  # Linear kernel
#yhat2 = ...  # Non-linear kernel

# Compute AUC
auc1 = sklearn.metrics.roc_auc_score(y_test, yhat1)
#auc2 = ...

print(auc1)
#print(auc2)
