import csv
import scipy.io as sio
import numpy as np
import scipy
# import matplotlib.pyplot as plt
import collections
import pdb
import time
from joblib import Parallel, delayed
import multiprocessing
# import seaborn as sns

D = 5
K = 5
REG = 0.1

def load_joke_dataset():
    filename = "joke_data/joke_train.mat"
    data = sio.loadmat(filename)
    train_data = data['train'].T

    val = []
    with open('joke_data/validation.txt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            val.append(list(map(int, row)))
    val = np.array(val)

    test = []
    with open('joke_data/query.txt', newline = "") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            test.append(list(map(int, row)))
    test = np.array(test)[:,1:3]
    return train_data, val, test

def simplePCA(train_data, d = D):
    train_data = np.nan_to_num(train_data)
    u, s, vt = scipy.sparse.linalg.svds(train_data, d)
    s = np.diag(s)
    pred_data = u.dot(s).dot(vt)
    return pred_data

def validate(pred_data, val):
    correct = 0
    for row in val:
        if ((row[2] == 1 and pred_data[row[1] - 1,row[0] - 1] > 0) or (row[2] == 0 and pred_data[row[1] - 1, row[0] - 1] <= 0)):
            correct += 1
    return correct / val.shape[0]

def test(pred_data, test):
    labels = []
    for row in test:
        if (pred_data[row[1] - 1, row[0] - 1] > 0):
            labels.append(1)
        else:
            labels.append(0)
    return labels

def min_err(train_data, d = D, reg = REG):
    u = np.rand((100, d))
    v = np.rand((d, 24983))

def load_mnist_dataset():

    mnist_data = sio.loadmat("mnist_data/images.mat")["images"]
    return mnist_data.reshape((28*28, 60000)).T


def main():
    train_data, val, test_users = load_joke_dataset()
    pred_data = simplePCA(train_data)
    val_acc = validate(pred_data, val)
    print("Validation Accuracy: {0}".format(val_acc))

    test_pred = test(pred_data, test_users)
    id = np.array(list(range(1,len(test_pred) + 1)))
    output = np.array([id,test_pred]).T
    pdb.set_trace()
    np.savetxt("kaggle_submission_v1.csv", output, delimiter = ',')

if __name__ == "__main__":
    main()
