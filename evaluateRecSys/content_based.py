
from getData import *
import sqlite3
from sqlite3 import Error
from userItem import *
import numpy as np
from recommendations import *
from evaluationMetrics import *
import sys
import os
from getUserSimilarity import *
import random_Rec
import gc
import configargparse
import random

def content_based_0(simPathDir, userClusterPath, k):
    matrix = getUserClustersMatrixFromCSV(userClusterPath)
    originalIDs = matrix.index.values.tolist()
    dirList = os.listdir(simPathDir)
    limitCount = 0

    for file in dirList:
        print "similarity type: ", file

        simPath = simPathDir + file

        xSize, ySize = matrix.shape

        limit1 = int(ySize * 0.5)

        matrixReduced = reduceDataFrame(matrix, [0, 1])  # drop id and name

        sim = getSimilarityCSV(simPath)

        trainingSet, testSet = divideDataSet(matrixReduced, 0.5)

        totalPrecisions = []
        totalRecall = []

        count = 0
        for index, i in trainingSet.iterrows():
            print index
            userTrain = getOneUser(trainingSet, index)

            userTest = getOneUser(testSet, index)

            userNameTrain = matrix.user[index]

            # userNameTrain = getUserName(userTrain)

            numberOfClustersTrain = numberOfClustersPerUser(userTrain)

            numberOfClustersTest = numberOfClustersPerUser(userTest)
            print numberOfClustersTest

            if len(numberOfClustersTrain) == 0:
                # print "information about clusters for user #", userNameTrain, "# is insufficient!"
                continue

            if len(numberOfClustersTest) == 0:
                # print "information about clusters for user #", userNameTrain, "# is insufficient!"
                continue

            topkList = pd.DataFrame()
            for clust in numberOfClustersTrain:
                topk = getTopKTestSet(clust, k, sim, limit1)

                topkList = topkList.append(topk)

            # TODO: how to combine the similarities of each cluster?

            rec = getRecommendations(topkList)

            rec = np.unique(rec)

            totalRel = numberOfClustersTest
            # totalRel = getTotalRelevantForUser(numberOfClusters, 1)

            totalRelFromAlgoritm = relevantItems(rec, totalRel)

            userPrecision = precision(len(totalRelFromAlgoritm),
                                      len(rec))  # second arg should be k, but for now it is variable
            print "precision: ", userPrecision
            totalPrecisions.append(userPrecision)

            userRecall = recall(len(totalRelFromAlgoritm), len(totalRel))
            print "recal: ", userRecall
            totalRecall.append(userRecall)

            count += 1

        finalP = precisionMean(totalPrecisions, len(totalPrecisions))
        finalR = recallMean(totalRecall, len(totalRecall))

        print file
        print "recommender precision at k =", k, " = ", finalP
        print "recommender recall at k =", k, " = ", finalR
        print len(totalPrecisions), " users"

        totalRecall = []
        totalPrecisions = []



def content_based_1(simPathDir, userClusterPath, k):

    dirList = os.listdir(simPathDir)
    limitCount = 0
    for file in dirList:

        simPath = simPathDir + file

        matrix = getUserClustersMatrixFromCSV(userClusterPath)
        sim = getSimilarityCSV(simPath)

        totalPrecisions = []
        totalRecall = []

        count = 0
        for index, i in matrix.iterrows():
            userTest = getOneUser(matrix, index)
            userName = getUserName(userTest)
            numberOfClusters = numberOfClustersPerUser(userTest)

            if len(numberOfClusters) < 2:
                print "information about clusters for user #", userName, "# is insufficient!"
                continue

            topk = getTopK(numberOfClusters[0], k, sim)  # just passing the first clusters

            rec = getRecommendations(topk)

            totalRel = getTotalRelevantForUser(numberOfClusters, 1)

            totalRelFromAlgoritm = relevantItems(rec, totalRel)

            userPrecision = precision(len(totalRelFromAlgoritm), k)

            totalPrecisions.append(userPrecision)

            userRecall = recall(len(totalRelFromAlgoritm), len(totalRel))

            totalRecall.append(userRecall)

            count += 1

        finalP = precisionMean(totalPrecisions, len(totalPrecisions))
        finalR = recallMean(totalRecall, len(totalRecall))

        print file
        print "recommender precision at k =", k, " = ", finalP
        print "recommender precision at k =", k, " = ", finalR
        print len(totalPrecisions), " users"


def content_based_rattings(simPathDir, matrix, k, nCV, randUers, ktop):

    print "content-based cross validation RMSE @ ", k, "top k @ ", ktop



    dirList = os.listdir(simPathDir)
    limitCount = 0
    for file in dirList:

        RMSEList = []
        precList = []
        recList = []

        print file
        simPath = simPathDir + file

        sim = getSimilarityCSV(simPath)

        limitMin = 0
        limitMax = 0

        crossCount = 0

        for i in xrange(nCV):
            print "i ", i

            print matrix.shape
            print matrix.max().max()

            xSize, ySize = matrix.shape

            if crossCount == nCV - 1:
                limitMax = ySize
            else:
                limitMax = limitMin + (int(ySize / nCV))

            print "min: ", limitMin

            print "max: ", limitMax

            trainingSet, testSet = divideDataSetCrossValidation(matrix, limitMin, limitMax)
            #trainingSet = trainingSet.loc[(trainingSet != 0).any(1)]


            trainingSet, testSet = excludeTrainAndTestAllZeros(trainingSet, testSet)


            if randUers == 0:
                randUers = len(trainingSet)

            originalIDs = random.sample(trainingSet.index, randUers)
            # originalIDs = random.sample(range(0, xSize), 5)
            # originalIDs = [1929, 5432, 11096, 13671, 14749]

            originalIDs = sorted(originalIDs)

            trainingSet = trainingSet.ix[originalIDs]
            testSet = testSet.ix[originalIDs]


            trainingClusters = (trainingSet.columns.values).astype(np.int)

            testClusters = (testSet.columns.values).astype(np.int)

            y_pred = []
            print testClusters.shape
            for cluster in testClusters:
                #print cluster
                topk = getTopK(cluster, k, sim, testClusters)

                rec = getRecommendations(topk)


                meanRatings = getMeanRatingsCB(trainingSet, rec)

                y_pred.append(meanRatings.tolist())

            y_pred = pd.DataFrame(y_pred).transpose()
            y_pred = y_pred.dropna(axis='columns')
            indexes = y_pred.columns.values.tolist()


            testSet = pd.DataFrame(np.array(testSet))[indexes]

            y_pred, y_real = deleteAllZerosTestSet(y_pred, testSet)
            RMSE = calculateRMSE(y_real, y_pred)

            print np.array(y_pred).shape
            print "RMSE = ", RMSE

            rankReal, rankPred = getTopKRatings(y_pred, y_real, ktop)

            relevantsInTopK = getnRelevants(rankReal)
            totalRelevants = getnRelevants(y_real)

            allPrecisions = precision(relevantsInTopK, ktop)
            allRecalls = recall(relevantsInTopK, totalRelevants)
            print "Recall = ", allRecalls.mean()
            print "Precision = ", allPrecisions.mean()

            RMSEList.append(RMSE)
            precList.append(allPrecisions.mean())
            recList.append(allRecalls.mean())

            gc.collect()

            limitMin = limitMax
            crossCount += 1

            randUers = 0

        print "RMSE CV mean = ", np.array(RMSEList).mean()
        print "Recall CV mean = ", np.array(recList).mean()
        print "Precision CV mean = ", np.array(precList).mean()




