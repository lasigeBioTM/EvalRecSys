from getData import *
from userItem import *
from recommendations import *
from evaluationMetrics import *
import sys
import os
from getUserSimilarity import *
import random
import gc
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



def collaborative_filtering_2(userClusterPath, randUers, similarityMeasure, kCF):
    for i in xrange(5):
        print "i ", i
        # collaborative filtering
        matrix = getUserClustersMatrixFromCSV(userClusterPath)

        matrixReduced = reduceDataFrame(matrix, [0, 1])  # drop the first two columns

        matrixReduced = normalizeDatasetToOnes(matrixReduced)

        trainingSet, testSet = divideDataSet(matrixReduced, 0.5)

        trainingSet = trainingSet.loc[(trainingSet != 0).any(1)]

        recommendations = []

        xSize, ySize = matrixReduced.shape

        if randUers == 0:
            randUers = len(trainingSet)

        originalIDs = random.sample(trainingSet.index, randUers)
        # originalIDs = random.sample(range(0, xSize), 5)
        # originalIDs = [1929, 5432, 11096, 13671, 14749]

        originalIDs = sorted(originalIDs)
        trainingSet = trainingSet.ix[originalIDs]

        # trainingSet = np.array(trainingSet)[originalIDs]
        # print trainingSet.shape
        progressionCount = 0

        for userID in originalIDs:

            print "progression: ", (float(progressionCount) / len(originalIDs)) * 100, " %"
            sys.stdout.flush()

            userSim = getUserSimilarity(trainingSet, similarityMeasure, originalIDs, userID)

            np.set_printoptions(suppress=True)

            mostSimilarUSer = getMostSimilarCF(userSim, userID, kCF)

            similarUsersID = getMostSimilarUserID(mostSimilarUSer)

            for userRecID in similarUsersID:
                recommendations.append(testSet.ix[userRecID].tolist())

                # print np.where(np.array(testSet)[userID] > 0)
                # print np.where(np.array(testSet)[userRecID] > 0)

            progressionCount += 1

        testSetToEvaluate = testSet.ix[originalIDs]
        print testSetToEvaluate.shape

        testSetToEvaluate = np.array(testSetToEvaluate)

        recommendations = np.array(recommendations)
        print recommendations.shape

        new_train, new_test = deleteAllZerosTestSet(recommendations, testSetToEvaluate)

        RMSE = calculateRMSE(new_test, new_train)
        print RMSE
        gc.collect()

        # for rec, test in zip(recommendations, testSetToEvaluate):
        #     print np.where(rec > 0)
        #     print np.where(test > 0)
        #
        #     print len(test)
        #     print (rec == test).sum()

def collaborative_filtering_user_user(matrix, randUers, similarityMeasure, kCF, nCV, topk):
    print "user based collaborative filtering cross validation RMSE @ ", kCF, "top k @ ", topk
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

        #matrixReduced = normalizeDatasetToOnes(matrixReduced)
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


        y_pred = []

        # len(trainingSet)

        if randUers == 0:
            randUers = len(trainingSet)


        print "n users being analysed = ", randUers
        originalIDs = random.sample(trainingSet.index, randUers)
        # originalIDs = random.sample(range(0, xSize), 5)
        # originalIDs = [1929, 5432, 11096, 13671, 14749]

        originalIDs = sorted(originalIDs)
        trainingSet = trainingSet.ix[originalIDs]

        # trainingSet = np.array(trainingSet)[originalIDs]
        # print trainingSet.shape
        progressionCount = 0

        for userID in originalIDs:

            #print "progression: ", i, "_", (float(progressionCount) / len(originalIDs)) * 100, " %"

            sys.stdout.flush()

            userSim = getUserSimilarity(trainingSet, similarityMeasure, originalIDs, userID)

            np.set_printoptions(suppress=True)

            mostSimilarUSer = getMostSimilarCF(userSim, userID, kCF)

            similarUsersID, simValues = getMostSimilarUserID(mostSimilarUSer)


            meanRatting = calculateMeankCF(testSet, similarUsersID, simValues)

            y_pred.append(meanRatting)
            # for userRecID in similarUsersID:
            #     recommendations.append(testSet.ix[userRecID].tolist())

                # print np.where(np.array(testSet)[userID] > 0)
                # print np.where(np.array(testSet)[userRecID] > 0)

            progressionCount += 1


        testSetToEvaluate = testSet.ix[originalIDs]

        y_real = np.array(testSetToEvaluate)

        y_pred = np.array(y_pred)

        y_pred, y_real = deleteAllZerosTestSet(y_pred, y_real)

        RMSE = calculateRMSE(y_real, y_pred)
        print "RMSE = ", RMSE
        rankReal, rankPred = getTopKRatings(y_pred, y_real, topk)


        relevantsInTopK = getnRelevants(rankReal)
        totalRelevants = getnRelevants(y_real)


        allPrecisions = precision(relevantsInTopK, topk)
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




def predict(ratings, similarity, type='user'):
    if type == 'user':
        print "user"
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        print "item "
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


    return pred



def collaborative_filtering_6_item_item(matrix, randUers, similarityMeasure, kCF, nCV, topk):
    print "item based collaborative filtering cross validation RMSE @ ", kCF, "top k @ ", topk

    RMSEList = []
    precList = []
    recList = []

    #trainSet, testSet = train_test_split(matrix, test_size=0.25)

    kf = KFold(n_splits=5)
    kf.get_n_splits(matrix)

    for train_index, test_index in kf.split(matrix):
        print test_index
        trainSet = matrix.iloc[train_index]
        testSet = matrix.iloc[test_index]


        #trainSet = matrix.reindex(train_index)#[train_index]
        #testSet = matrix.reindex(test_index)#[test_index]


        trainSet = trainSet.T
        testSet = testSet.T

        trainSet, testSet = excludeTrainAndTestAllZeros(trainSet, testSet)


        itemsIDs = trainSet.index

        itemIDsTest = testSet.index

        xTrain, yTrain = trainSet.shape


        similaritiesMatrix = np.zeros((xTrain, xTrain))


        xTest, yTest = testSet.shape

        #meanRatings = np.zeros((xTest, yTest))
        y_pred = []

        for id in itemsIDs:
            #print id
            item = np.array(trainSet.ix[id])
            item = np.reshape(item, (-1, yTrain))

            itemSim = getItemItemSimilarity(item, np.array(trainSet), similarityMeasure, itemsIDs, id)

            mostSimilarItems = getMostSimilarCF(itemSim, id, kCF)

            similarItemsID, simValues = getMostSimilarUserID(mostSimilarItems)

            meanRatting = calculateMeankCF(testSet, similarItemsID, simValues)

            #similaritiesMatrix[int(id)-1, (similarItemsID.astype(int) -1).tolist()] = simValues
            #similaritiesMatrix[(similarItemsID.astype(int) -1).tolist(), int(id)-1] = simValues

            y_pred.append(meanRatting)

        y_real = np.array(testSet)

        y_pred = np.array(y_pred)

        y_pred, y_real = deleteAllZerosTestSet(y_pred, y_real)

        RMSE = calculateRMSE(y_real.T, y_pred.T)
        print "RMSE = ", RMSE
        rankReal, rankPred = getTopKRatings(y_pred.T, y_real.T, topk)

        relevantsInTopK = getnRelevants(rankReal)
        totalRelevants = getnRelevants(y_real.T)

        allPrecisions = precision(relevantsInTopK, topk)
        allRecalls = recall(relevantsInTopK, totalRelevants)
        print "Recall = ", allRecalls.mean()
        print "Precision = ", allPrecisions.mean()

        RMSEList.append(RMSE)
        precList.append(allPrecisions.mean())
        recList.append(allRecalls.mean())

    print "RMSE CV mean = ", np.array(RMSEList).mean()
    print "Recall CV mean = ", np.array(recList).mean()
    print "Precision CV mean = ", np.array(precList).mean()


def collaborative_filtering_MF(matrix, nCV, topk):
    print "MF collaborative filtering cross validation top k @ ", topk

    RMSEList = []
    precList = []
    recList = []

    matrix = np.array(matrix)
    matrix = matrix.astype(float)
    xSize, ySize = matrix.shape

    limitMin = 0
    limitMax = 0

    crossCount = 0

    for i in xrange(nCV):
        print "i ", i

        if crossCount == nCV - 1:
            limitMax = ySize
        else:
            limitMax = limitMin + (int(ySize / nCV))

        #print "min: ", limitMin

        #print "max: ", limitMax


        testSet = matrix[:, limitMin:limitMax]

        #delete columns from limit min, to limit max, exclusive
        trainSet = np.delete(matrix, np.s_[limitMin:limitMax], axis=1)
        trainSet, testSet = excludeTrainAndTestAllZeros(trainSet, testSet)
        print trainSet.shape
        print testSet.shape

        xTest, yTest = testSet.shape

        zerosArray = np.zeros((xTest, yTest))

        newMatrix = np.concatenate((trainSet, zerosArray), axis=1)


        #newMatrix = csc_matrix(newMatrix, dtype=float)


        #ks = np.arange(1, ySize, 1)

        ks = [1325]

        for a in ks:

            newMatrix = csc_matrix(newMatrix, dtype=float)
            u, s, vt = svds(newMatrix, k=a)


            s_diag_matrix = np.diag(s)

            X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

            y_pred = X_pred[:, limitMin:limitMax]

            y_pred, testSet = deleteAllZerosTestSet(y_pred, testSet)
            #print y_pred.shape
            #print testSet.shape
            RMSE = calculateRMSE(testSet, y_pred)


            #print RMSE
            rankReal, rankPred = getTopKRatings(y_pred, testSet, topk)

            relevantsInTopK = getnRelevants(rankReal)
            totalRelevants = getnRelevants(testSet)

            allPrecisions = precision(relevantsInTopK, topk)
            allRecalls = recall(relevantsInTopK, totalRelevants)
            #print allRecalls.mean()
            #print allPrecisions.mean()

            print "RMSE = ", RMSE

            print "Recall = ", allRecalls.mean()
            print "Precision = ", allPrecisions.mean()

            RMSEList.append(RMSE)
            precList.append(allPrecisions.mean())
            recList.append(allRecalls.mean())

            gc.collect()

        limitMin = limitMax
        crossCount += 1

    print "RMSE CV mean = ", np.array(RMSEList).mean()
    print "Recall CV mean = ", np.array(recList).mean()
    print "Precision CV mean = ", np.array(precList).mean()



