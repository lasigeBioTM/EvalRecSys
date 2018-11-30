
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
from collaborative_filtering import *
from content_based import *
from random_Rec import *



def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


if __name__ == '__main__':

    p = configargparse.ArgParser(default_config_files=['../config.ini'])
    p.add('-mc', '--my-config', is_config_file=True, help='alternative config file path')

    p.add("-userItem", "--user_itemPath", required=False, help="path to user_item matrix", type=str)

    p.add("-db", "--db_file", required=True, help="path do database",
          type=str)

    p.add("-simdir", "--simPathDir", required=True, help="path to directory with clusters similarities",
          type=str)

    p.add("-k", "--k", required=False, help="top-k similar clusters for CB",
          type=int)

    p.add("-kCF", "--kCF", required=False, help="top-k similar users for CF",
          type=int)

    p.add("-alg", "--algorithm", required=False, help="algorithm for evaluating",
          type=int)

    p.add("-simMeasCF", "--similarityMeasureCF", required=False, help="similarity measure to use with CF",
          type=str)

    p.add("-simMeasCB", "--similarityMeasureCB", required=False, help="similarity measure to use with CB",
          type=str)
    p.add("-param", "--parameters", required=False, help="list of parameters to calculate similarity between items",
          type=str)

    p.add("-nCV", "--n_crossValidation", required=False, help="number of dataset divition for cross-validation",
          type=int)

    p.add("-randUsers", "--numberOfRandomUsers", required=False,
          help="size of random users to give the recommendations",
          type=int)

    p.add("-topk", "--topk", required=False,
          help="topk",
          type=int)

    options = p.parse_args()

    userClusterPath = options.user_itemPath
    dataBasePath = options.db_file
    simPathDir = options.simPathDir
    k = options.k
    kCF = options.kCF
    dataSetDivision = options.algorithm

    similarityMeasure = options.similarityMeasureCF

    nCV = options.n_crossValidation

    randUers = options.numberOfRandomUsers

    topk = options.topk

    matrix = getUserClustersMatrixFromCSV(userClusterPath)
    matrixReduced = reduceDataFrame(matrix, [0, 1])  # drop the first two columns
    matrixNormalized = normalizePandasDF(matrixReduced)
    matrixCleanned = excludeUersUniqueCluster(matrixNormalized)



    if dataSetDivision == 0: #do not use
        content_based_0(simPathDir, userClusterPath, k)

    if dataSetDivision == 1: #do not use
        content_based_1(simPathDir, userClusterPath, k)

    if dataSetDivision == 2: #random users #do not use
        collaborative_filtering_2(userClusterPath, randUers, similarityMeasure, kCF)



    if dataSetDivision == 3: #cross validation
        collaborative_filtering_user_user(matrixCleanned, randUers, similarityMeasure, kCF, nCV, topk)

    if dataSetDivision == 4:
        random_rec(matrixCleanned, nCV, topk)

    if dataSetDivision == 5:
        content_based_rattings(simPathDir, matrixCleanned, k, nCV, randUers, topk)

    if dataSetDivision == 6:
        collaborative_filtering_6_item_item(matrixCleanned, randUers, similarityMeasure, kCF, nCV, topk)

    if dataSetDivision == 7:
        collaborative_filtering_MF(matrixCleanned, nCV, topk)
