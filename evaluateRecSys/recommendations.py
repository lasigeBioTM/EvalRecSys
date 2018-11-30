import numpy as np
import sys


def getMeanRatingsCB(trainingSet, rec):
    '''
    get mean rating a user gave for the most similar clusters to the one being analysed
    :param trainingSet: pandas DataFrame training set user_item
    :param rec: numpy array with the id of the most similar clusters
    :return: mean of ratings of the clusters in rec
    '''

    mean = trainingSet[rec.astype(str).tolist()].mean(axis=1)

    return mean


def getMostSimilarUserID(similarities):
    '''
    pandas top k similarities to array numpy
    :param recommendations: pandas top k similarities
    :return: numpy array of most similar users ids
    '''

    usersIDs = np.array(similarities[1])
    simValues = np.array(similarities[2])

    return usersIDs, simValues



def calculateMeankCFii(testSet, similarItemsID, simValues):
    '''
      calculate the mean ratings for k using, using the values of similarities to weight
      :param testSet: dataset of tests
      :param similarUsersID: array with ids of similar users
      :param simValues: array with similarities
      :return: array with weighted similarities
      '''


    testSet_numpy = np.array(testSet.ix[similarItemsID])

    usersMEan = np.mean(((simValues * testSet_numpy.T).T),axis=0).tolist()

    #usersMEan = testSet.ix[similarUsersID].mean(axis=0).tolist()


    return usersMEan


def calculateMeankCFii(testSet, similarItemsID, simValues):
    '''
      calculate the mean ratings for k using, using the values of similarities to weight
      :param testSet: dataset of tests
      :param similarUsersID: array with ids of similar users
      :param simValues: array with similarities
      :return: array with weighted similarities
      '''

    print testSet
    print similarItemsID


    testSet_numpy = np.array(testSet.ix[similarItemsID])


    usersMEan = np.mean(((simValues * testSet_numpy.T).T),axis=0).tolist()

    #usersMEan = testSet.ix[similarUsersID].mean(axis=0).tolist()


    return usersMEan


def calculateMeankCF(testSet, similarUsersID, simValues):
    '''
      calculate the mean ratings for k using, using the values of similarities to weight
      :param testSet: dataset of tests
      :param similarUsersID: array with ids of similar users
      :param simValues: array with similarities
      :return: array with weighted similarities
      '''


    testSet_numpy = np.array(testSet.ix[similarUsersID])

    usersMEan = np.mean(((simValues * testSet_numpy.T).T),axis=0).tolist()

    #usersMEan = testSet.ix[similarUsersID].mean(axis=0).tolist()


    return usersMEan


def getMostSimilarCFItemItem(sim, kCF):

    '''
    get the most similar items for one item
    :param sim: pd DataFrame Of similarities for one item
    :param userID: user ID being considered
    :param kCF: number of most similars users desired
    :return: pd dataframe of kCF most similares users to userID
    '''

    simToClust = sim.sort_values(by=[2], ascending=False)


    #simToClust = simToClust[~((simToClust[0] == userID) & (simToClust[1] == userID))]

    mostSimilarItems = simToClust.head(kCF)

    itemsIDs = np.array(mostSimilarItems[1])
    simValues = np.array(mostSimilarItems[2])


    return itemsIDs, simValues



def getMostSimilarCF(sim, userID, kCF):

    '''
    get the most similar users for one user
    :param sim: pd DataFrame Of similarities for one user
    :param userID: user ID being considered
    :param kCF: number of most similars users desired
    :return: pd dataframe of kCF most similares users to userID
    '''

    simToClust = sim.sort_values(by=[2], ascending=False)


    simToClust = simToClust[~((simToClust[0] == userID) & (simToClust[1] == userID))]

    mostSimilarUsers = simToClust.head(kCF)


    return mostSimilarUsers


def getTopKTestSet(clustID, k, similarities, limit1):
    '''
    get the first top k similarities for a user
    considering one clusters
    :param clustID: if of the cluster
    :param k: number maximum of similarities
    :param similarities: pandas df with the similariries
    :return: pandas df with top k similarties for a cluster
    '''

    simToClust = similarities[(similarities[0] == clustID)].sort_values(by=[2],ascending=False)

    simToClust = simToClust[~((simToClust[0] == clustID) & (simToClust[1] == clustID))]

    simToClust = simToClust.drop(simToClust[simToClust[1] < limit1].index)

    recommendations = simToClust.head(k)

    return recommendations


def getTopK(clustID, k, similarities, toDrop):
    '''
    get the first top k similarities for a user
    considering one clusters
    :param clustID: if of the cluster
    :param k: number maximum of similarities
    :param similarities: pandas df with the similariries
    :return: pandas df with top k similarties for a cluster
    '''


    simToClust = similarities[(similarities[0] == clustID)].sort_values(by=[2],ascending=False)

    simToClust = simToClust[~((simToClust[0] == clustID) & (simToClust[1] == clustID))]

    #drop clusters ids in the testSet

    simToClust = simToClust[~simToClust[1].isin(toDrop)]

    recommendations = simToClust.head(k)

    return recommendations


def relevantItems(recommendation, totalOfRelevants):
    '''
    from the recommended items, which were right recommended
    :param recommendation: numpy array of top k recommendations
    :param totalOfRelevants: numpy array of total relevant items for user
    :return: array of relevant items correctly recommended by the algorithm
    '''

    totalRelevant = np.intersect1d(recommendation, totalOfRelevants)

    return totalRelevant


def getRecommendations(recommendations):
    '''
    pandas top k similarities to array numpy
    :param recommendations: pandas top k similarities
    :return: numpy array of similarities
    '''

    recommendations = np.array(recommendations[1])

    return recommendations


def getTotalRelevantForUser(allClusters, n):
    totalRelevant = allClusters[n:]

    return totalRelevant
