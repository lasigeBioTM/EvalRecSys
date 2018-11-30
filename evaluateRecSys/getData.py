import pandas as pd

def getClustersFromDB(conn):
    '''
    easy function to get data from sqlite db
    :param conn: connection to database
    :return: pandas dataframe with clusters table
    '''

    #"SELECT * FROM clusters WHERE simbadID is not NULL"
    df = pd.read_sql_query("SELECT * FROM clusters", conn)

    return df

def getUserClustersMatrixFromCSV(csvPath):

    matrix = pd.read_csv(csvPath)

    return pd.DataFrame(matrix)


def getSimilarityCSV(csvPath):

    sim = pd.read_csv(csvPath,header=None)

    return pd.DataFrame(sim)