from getData import *
from userItem import *
from recommendations import *
from evaluationMetrics import *
import sys
import os
from getUserSimilarity import *
import random_Rec
import gc


def random_rec(matrix, nCV, topk):

    print "Random top k @ ", topk

    RMSEList = []
    precList = []
    recList = []

    limitMin = 0
    limitMax = 0

    crossCount = 0

    for i in xrange(nCV):
        print "i ", i
        # collaborative filtering

        print matrix.shape

        # matrixReduced = normalizeDatasetToOnes(matrixReduced)
        xSize, ySize = matrix.shape

        if crossCount == nCV - 1:
            limitMax = ySize
        else:
            limitMax = limitMin + (int(ySize / nCV))

        print "min: ", limitMin

        print "max: ", limitMax

        trainingSet, testSet = divideDataSetCrossValidation(matrix, limitMin, limitMax)
        # trainingSet = trainingSet.loc[(trainingSet != 0).any(1)]

        trainingSet, testSet = excludeTrainAndTestAllZeros(trainingSet, testSet)


        print testSet.shape

        xTest, yTest = testSet.shape

        recommendations = np.random.uniform(low=0.0, high=1, size=(xTest, yTest))
        print recommendations.shape

        RMSE = calculateRMSE(testSet, recommendations)
        print RMSE

        new_test = np.array(testSet)
        rankReal, rankPred = getTopKRatings(recommendations, new_test, topk)

        relevantsInTopK = getnRelevants(rankReal)
        totalRelevants = getnRelevants(new_test)

        allPrecisions = precision(relevantsInTopK, topk)
        allRecalls = recall(relevantsInTopK, totalRelevants)
        print allRecalls.mean()
        print allPrecisions.mean()

        RMSEList.append(RMSE)
        precList.append(allPrecisions.mean())
        recList.append(allRecalls.mean())

        gc.collect()

        limitMin = limitMax
        crossCount += 1

    print "RMSE CV mean = ", np.array(RMSEList).mean()
    print "Recall CV mean = ", np.array(recList).mean()
    print "Precision CV mean = ", np.array(precList).mean()
