import random
import numpy as np
from cs231n.data_utils import load_CIFAR10

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape

from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = []

#for f in xrange(num_folds):
#    X_train_val = np.concatenate([j for i,j in enumerate(X_train_folds) if i!=f])
#    y_train_val = np.concatenate([j for i,j in enumerate(y_train_folds) if i!=f])

X_train_val = np.concatenate([j for i,j in enumerate(X_train_folds) if i!=0])
y_train_val = np.concatenate([j for i,j in enumerate(y_train_folds) if i!=0])

    classifier.train(X_train_val, y_train_val)

    for k in k_choices:
        y_pred = classifier.predict(X_train_folds[f], k)
        
        num_correct = np.sum(y_pred == y_train_folds[f])
        accuracy = float(num_correct) / float(y_train_folds[f].shape[0])
        k_to_accuracies[k].append(accuracy)


