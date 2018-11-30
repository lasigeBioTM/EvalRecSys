import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


def getTopKRatings(y_pred, y_actual, k):

    totalTopReal = []
    totalTopPred = []

    for pred, actual in zip(y_pred, y_actual):

        #sorted_pred = -np.sort(-pred)

        sorted_pred = np.sort(pred)
        sorted_pred = sorted_pred[::-1]

        #indexes_sorted_pred = np.argsort(-pred)
        indexes_sorted_pred = np.argsort(pred)
        indexes_sorted_pred = indexes_sorted_pred[::-1]

        indices = range(k)

        topk = sorted_pred[indices]
        topk_indeces = indexes_sorted_pred[indices]

        topActual = actual[topk_indeces]

        totalTopPred.append(topk.tolist())
        totalTopReal.append(topActual.tolist())

        #print "pred: ", topk
        #print "real: ", topActual



    return np.array(totalTopReal), np.array(totalTopPred)



def deleteAllZerosTestSet(trainSet, testSet):
    testSet = pd.DataFrame(testSet)

    trainSet = pd.DataFrame(trainSet)

    testSet1 = testSet.loc[(testSet != 0).any(1)]

    trainSet = trainSet[(testSet != 0).any(1)]

    return np.array(trainSet), np.array(testSet1)


def calculateRMSE(y_actual, y_predicted):
    #y_predicted = y_predicted[np.where(y_actual > 0)]
    #y_actual = y_actual[np.where(y_actual>0)]

    rms = sqrt(mean_squared_error(y_actual, y_predicted))

    return rms


def precision(rightRec,totalRec):

    precision = rightRec.astype(float)/totalRec

    return precision


def precisionMean(precision, n):
    if n==0:
        pMean = 1

    else:
        pMean = np.sum(np.array(precision))/n

    return pMean


def recall(rightRec, totalRel):
    recall = rightRec.astype(float)/totalRel.astype(float)

    return recall


def recallMean(recall, n):
    if n == 0:
        rMean = 1

    else:
        rMean = np.sum(np.array(recall)) / n

    return rMean

def getnRelevants(array):

    relevants = []

    for row in array:

        nRelevant = len(row[np.where(row > 0.0)])
        relevants.append(nRelevant)

    return np.array(relevants)
