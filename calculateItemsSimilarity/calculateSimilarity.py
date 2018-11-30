import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from sklearn import preprocessing
from scipy import sparse
import pandas as pd


def reshapeSizeOneArray(array):
    zeroArray = np.ones((len(array),1))
    array = np.hstack(((array,zeroArray)))

    return array


def dealingwithNan(df):
    '''
    replace nan with zero
    :param df: pandas dataframe
    :return: pandas dataframe
    '''
    df = df.astype(float).fillna(0.0)

    return df


def getSimilarityMatrixCdist(array1, array2,simType):
    '''
    uses cosine similarity to calculate similarity between rows in a dataset
    :param df: normalized array
    :return: numpy array of similarities mSourcesxMsources
    '''

    similarities = 1 - cdist(array1, array2, metric=simType) #pdist(df, metric='cosine') = distance; 1-pdist(df, metric='cosine') = similarity

    if simType=='euclidean':
        similarities = (similarities-np.amin(similarities))/(np.amax(similarities)-np.amin(similarities))

    #similarities = squareform(similarities)


    return similarities

def getSimilarityMatrix(df,simType):
    '''
    uses cosine similarity to calculate similarity between rows in a dataset
    :param df: normalized array
    :return: numpy array of similarities mSourcesxMsources
    '''

    similarities = 1 - pdist(df, metric=simType) #pdist(df, metric='cosine') = distance; 1-pdist(df, metric='cosine') = similarity
    # print np.amax(similarities)
    # print np.amin(similarities)
    # similarities = (similarities-np.amin(similarities))/(np.amax(similarities)-np.amin(similarities))

    if simType=='euclidean':
        similarities = (similarities-np.amin(similarities))/(np.amax(similarities)-np.amin(similarities))

    similarities = squareform(similarities)


    return similarities


def normalizeDataset(df):
    '''
    Normalize each column between zero and one,
    according to the max and min value of each column
    :param df: pandas dataframe
    :return: scaled numpy array
    '''

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)

    return x_scaled


def multiplyColumn(array, columnIndex, k):
    array[:, columnIndex] *= k

    return array

def getOneCluster(similarityMatrix, numberRow):

    oneCluster = similarityMatrix[numberRow]

    return oneCluster
