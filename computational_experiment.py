import sklearn.neighbors as neighbors
import sklearn.naive_bayes as naive_bayes
import sklearn.svm as svm
import sklearn.multiclass as multiclass
import sklearn.linear_model as linear_model
import sklearn.tree as tree
import sklearn.ensemble as ensemble

from sklearn.cross_validation import cross_val_score
from sklearn import grid_search
import sys
import numpy as np

file_name = sys.argv[1]

data = np.array([[0,0,0,0,0,0,0,0,0,0,0]])
target = np.array([])

file_in = open(file_name)
for line in file_in:
    sl = line.split(',')
    d = np.array([[float(sl[1]), float(sl[2]), float(sl[3]), float(sl[4]), float(sl[5]), float(sl[6]),
        float(sl[7]), float(sl[8]), float(sl[9]), float(sl[10]), float(sl[11])]])
    data = np.concatenate((data,d))
    target = np.append(target, sl[0])
data = data[1:,:]
file_in.close()


# define classifiers
kNeighborsClassifier = neighbors.KNeighborsClassifier()
nearestCentroid = neighbors.NearestCentroid()
gaussianNB = naive_bayes.GaussianNB()
multinomialNB = naive_bayes.MultinomialNB()
bernoulliNB = naive_bayes.BernoulliNB()
linearSVC = svm.LinearSVC()
oneVsRestClassifier = multiclass.OneVsRestClassifier(linearSVC)
oneVsOneClassifier = multiclass.OneVsOneClassifier(linearSVC)
ridgeClassifier = linear_model.RidgeClassifier()
logisticRegression = linear_model.LogisticRegression()
decisionTreeClassifier = tree.DecisionTreeClassifier()
extraTreeClassifier = tree.ExtraTreeClassifier()
extraTreesClassifier = ensemble.ExtraTreesClassifier()
adaBoost = ensemble.AdaBoostClassifier()
randomForest = ensemble.RandomForestClassifier()
baggingClassifier = ensemble.BaggingClassifier()
gradientBoostingClassifier = ensemble.GradientBoostingClassifier()

classifiers = [
    kNeighborsClassifier,
    nearestCentroid,
    gaussianNB,
    multinomialNB,
    bernoulliNB,
    linearSVC,
    oneVsRestClassifier,
    oneVsOneClassifier,
    ridgeClassifier,
    logisticRegression,
    decisionTreeClassifier,
    extraTreeClassifier,
    extraTreesClassifier,
    adaBoost,
    randomForest,
    baggingClassifier,
    gradientBoostingClassifier
]

classifiers_names = [
    "kNeighborsClassifier",
    "nearestCentroid",
    "gaussianNB",
    "multinomialNB",
    "bernoulliNB",
    "linearSVC",
    "oneVsRestClassifier",
    "oneVsOneClassifier",
    "ridgeClassifier",
    "logisticRegression",
    "decisionTreeClassifier",
    "extraTreeClassifier",
    "extraTreesClassifier",
    "adaBoost",
    "randomForest",
    "baggingClassifier",
    "gradientBoostingClassifier"
]

# Classifiers test
for classifier_nr in range(len(classifiers)):
    scores = cross_val_score(classifiers[classifier_nr], data, target, cv=5)
    print("{}: {}".format(classifiers_names[classifier_nr], scores.mean()))


# Features test
extraTreesClassifier = ensemble.ExtraTreesClassifier()
scores = []
n = []
for i in range(1,50):
    extraTreesClassifier.n_estimators = i
    n.append(i)
    scores.append(cross_val_score(extraTreesClassifier, data, target, cv=5).mean())
    print("n_estimators: ", n[-1], scores[-1])

extraTreesClassifier = ensemble.ExtraTreesClassifier()
scores = []
n = []
for i in range(1,12):
    extraTreesClassifier.max_features = i
    n.append(i)
    scores.append(cross_val_score(extraTreesClassifier, data, target, cv=5).mean())
    print("max_features: ", n[-1], scores[-1])

extraTreesClassifier = ensemble.ExtraTreesClassifier()
scores = []
n = []
for i in range(1,30):
    extraTreesClassifier.max_depth = i
    n.append(i)
    scores.append(cross_val_score(extraTreesClassifier, data, target, cv=5).mean())
    print("max_depth: ", n[-1], scores[-1])

extraTreesClassifier = ensemble.ExtraTreesClassifier()
scores = []
n = []
for i in range(2,30):
    extraTreesClassifier.min_samples_split = i
    n.append(i)
    scores.append(cross_val_score(extraTreesClassifier, data, target, cv=5).mean())
    print("min_samples_split: ", n[-1], scores[-1])

extraTreesClassifier = ensemble.ExtraTreesClassifier()
scores = []
n = []
for i in range(1,30):
    extraTreesClassifier.min_samples_leaf = i
    n.append(i)
    scores.append(cross_val_score(extraTreesClassifier, data, target, cv=5).mean())
    print("min_samples_leaf: ", n[-1], scores[-1])

extraTreesClassifier = ensemble.ExtraTreesClassifier()
scores = []
n = []
for i in range(2,30):
    extraTreesClassifier.max_leaf_nodes = i
    n.append(i)
    scores.append(cross_val_score(extraTreesClassifier, data, target, cv=5).mean())
    print("max_leaf_nodes: ", n[-1], scores[-1])


# Best params test
extraTreesClassifier = ensemble.ExtraTreesClassifier()
parameters = {'n_estimators': list(range(20, 35)), 'max_features': list(range(9, 12))}
clf = grid_search.GridSearchCV(extraTreesClassifier, parameters, cv=5)
clf.fit(data, target)
print(clf.best_estimator_)
print(clf.best_params_)
print(clf.best_score_)