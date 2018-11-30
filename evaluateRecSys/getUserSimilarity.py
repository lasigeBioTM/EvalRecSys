# matrix of users_item
# divide dataset
# normalizer dataset for only 0 or 1
# calculate similarities between users of trainning set
# save similarities

import pandas as pd
import numpy as np
import sys
from calculateSimilarity import *

def getUserClustersMatrixFromCSV(csvPath):
    matrix = pd.read_csv(csvPath)

    return pd.DataFrame(matrix)


def reduceDataFrame(df, toReduce):
    '''
    reduce data for the desired columns, to calculate similarity
    :param df: pandas df
    :return: pandas df reduced
    '''

    df = df.drop(df.columns[toReduce], axis=1)

    return df


def divideDataSetCrossValidation(df, limitMin, limitMax):
    '''

    :param df:
    :param limitMin:
    :param limitMax:
    :return:
    '''

    training = df.drop(df.columns[[np.arange(limitMin, limitMax)]], axis=1)

    test = df[df.columns[limitMin:limitMax]]
    #test = df.drop(df.columns[[np.arange(0, limit1)]], axis=1)

    return training, test


def divideDataSet(df, percentage):

    '''

    :param df:
    :param percentage:
    :return: pandas arrays
    '''
    sSize, ySize = df.shape

    limit1 = int(ySize * percentage)

    training = df.drop(df.columns[[np.arange(limit1, ySize)]], axis=1)

    test = df.drop(df.columns[[np.arange(0, limit1)]], axis=1)

    return training, test


def getItemItemSimilarity(item, trainingSet, similarityMeasure, listOfitems, clusterID):

    similarityMatrix = getSimilarityMatrixCdist(item, trainingSet, similarityMeasure)

    similaritiesForItem = []

    for id, sim in zip(listOfitems, similarityMatrix[0]):
        arraySim = [clusterID, id, sim]

        similaritiesForItem.append(arraySim)


    return pd.DataFrame(similaritiesForItem)


def getItemSimilarity(trainingSet, similarityMeasure, originalIDs, itemID):
    item = trainingSet.ix[itemID]

    trainingSet = np.array(trainingSet)
    # calculate similarities for training set

    xSize, ySize = trainingSet.shape

    similaritiesForItems = []
    # for user in trainingSet:

    item = np.reshape(item, (-1, ySize))
    similarityMatrix = getSimilarityMatrixCdist(item, trainingSet, similarityMeasure)

    for id, sim in zip(originalIDs, similarityMatrix[0]):
        arraySim = [itemID, id, sim]

        similaritiesForItems.append(arraySim)

    return pd.DataFrame(similaritiesForItems)


def getUserSimilarity(trainingSet, similarityMeasure, originalIDs, userID):
    user = trainingSet.ix[userID]


    trainingSet = np.array(trainingSet)
    # calculate similarities for training set

    xSize, ySize = trainingSet.shape


    similaritiesForUser = []
    count = 0
    # for user in trainingSet:

    user = np.reshape(np.array(user), (-1, ySize))
    similarityMatrix = getSimilarityMatrixCdist(user, trainingSet, similarityMeasure)

    for id, sim in zip(originalIDs, similarityMatrix[0]):
        arraySim = [userID, id, sim]

        similaritiesForUser.append(arraySim)

    return pd.DataFrame(similaritiesForUser)
