import numpy as np
import pandas as pd
from db_connection import *


def getAllClusters(conn):
    '''
    easy function to get data from sqlite db
    :param conn: connection to database
    :return: pandas dataframe with clusters table
    '''

    #"SELECT * FROM clusters WHERE simbadID is not NULL"
    df = pd.read_sql_query("SELECT * FROM clusters", conn)

    return df


def reduceDataFrame(df,toReduce):
    '''
    reduce data for the desired columns, to calculate similarity
    :param df: pandas df
    :return: pandas df reduced
    '''

    df = df.drop(df.columns[toReduce], axis=1)

    return df

def selectCoumns(df, listOfColumns):
    df = df[listOfColumns]

    return df

def getDataToSim(conn, listOfColumns):
    '''
    retrieve my data as a dataframe to calculate the similarities
    :param conn: connection to sqlite db
    :return: pandas dataframe
    '''
    df = getAllClusters(conn)

    df = selectCoumns(df, listOfColumns)

    return df

def getClustersName(df):

    names = df.name

    return names


def getClustersCoords(df, lonSys, latSys):

    lon = df[lonSys]
    lat = df[latSys]

    return lon, lat

#
# if __name__ == '__main__':
#     db_file = "/home/marcia/Desktop/dataBaseRecSys/adsInfoDBFull.db"
#     conn = create_connection(db_file)
#     clusters = getData(conn)


