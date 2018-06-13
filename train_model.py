#!/usr/bin/env python3

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
import numpy as np
import sys


def train_model():
    file_name = sys.argv[1]
    dump_file = "trained_model.pkl"
    print("Training for file", file_name)

    data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    target = np.array([])

    file_in = open(file_name)
    for line in file_in:
        sl = line.split(',')
        d = np.array([[float(sl[1]), float(sl[2]), float(sl[3]), float(sl[4]), float(sl[5]), float(sl[6]), float(sl[7]),
                       float(sl[8]), float(sl[9]), float(sl[10]), float(sl[11])]])
        data = np.concatenate((data, d))
        target = np.append(target, sl[0])
    data = data[1:, :]
    file_in.close()

    extraTreesClassifier = ExtraTreesClassifier(n_estimators=28, max_features=10)
    extraTreesClassifier.fit(data, target)
    joblib.dump(extraTreesClassifier, dump_file)
    print("Model saved as {}.".format(dump_file))


if __name__ == "__main__":
    train_model()
