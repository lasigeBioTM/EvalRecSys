import numpy as np



def excludeUersUniqueCluster(df):
    '''
    exclude users with only one item
    :param df: pandas df
    :return: pandas df
    '''

    mask = df.astype(bool).sum(axis=1) > 1
    df = df[mask]

    return df


def normalizeDatasetToOnes(df):
    '''
    change all values bigger than 1 to 1
    :param df: pandas dataframe
    :return: pandas dataframe tranformed
    '''

    df = df.mask(df > 1, 1)

    return df


def reduceDataFrame(df,toReduce):
    '''
    reduce data for the desired columns, to calculate similarity
    :param df: pandas df
    :return: pandas df reduced
    '''

    df = df.drop(df.columns[toReduce], axis=1)

    return df

def getOneUser(matrix, userID):
    '''
    get one row corresponding to a user(author)
    :param matrix: pandas dataframe
    :param userID: index of the row
    :return: one row pandas dataframe
    '''
    user = matrix.iloc[[userID]]



    return user


def getUserName(user):

    name = user.user

    return name.tolist()[0]


def numberOfClustersPerUser(user):
    '''

    :param user: pandas dataframe of a user
    :return: clusters the users "liked" (array numpy)
    '''

    #numberOfClusters = reduceDataFrame(user, [0, 1])
    numberOfClusters = user.where(user > 0)
    null_cols = numberOfClusters.columns[numberOfClusters.isnull().all()]
    numberOfClusters = numberOfClusters.drop(null_cols, axis=1)


    if numberOfClusters.size != 0:

         # drop id and name from user

        numberOfClusters = np.array(numberOfClusters.columns.astype(int))

    else:

        numberOfClusters = np.array([])


    return numberOfClusters


def divideDataSet(df, percentage):
    sSize, ySize = df.shape

    limit1 = int(ySize*percentage)

    dataSet1 = df.drop(df.columns[[np.arange(limit1, ySize)]], axis=1)

    dataSet2 = df.drop(df.columns[[np.arange(0,limit1)]], axis=1)


    return dataSet1, dataSet2


def userItemMatrixMean(matrix):


    return matrix


def normalizePandasDF(df):

    '''
    normalize pandas dataframe rows between 0 and 1,
    according to the max of each row
    :param df: pandas df
    :return: normalized pandas df
    '''
    df = df.divide(df.max(1), axis = 0)

    return df


def excludeTrainAndTestAllZeros(train, test):
    maskTrain = train.astype(bool).sum(axis=1) > 1

    #test = test[train.sum(1) > 0]
    test = test[maskTrain]

    train = train[maskTrain]

    #print train.sum(1) > 0

    maskTest = test.astype(bool).sum(axis=1) > 1
    train = train[maskTest]

    test = test[maskTest]


    return train, test