from calculateSimilarity import *
from getDataDB import *
import configargparse
import csv



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

    p.add("-alg", "--algorithm", required=False,
          type=int)

    p.add("-simMeasCF", "--similarityMeasureCF", required=False, help="similarity measure to use with CF",
          type=str)

    p.add("-simMeasCB", "--similarityMeasureCB", required=True, help="similarity measure to use with CB",
          type=str)
    p.add("-param", "--parameters", required=True, help="list of parameters to calculate similarity between items",
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
    db_file = options.db_file
    similarityMeasure = options.similarityMeasureCB
    simPathDir = options.simPathDir
    params = options.parameters

    conn = create_connection(db_file)
    normalize = True




    #get all data from the clusters dataset to calculate similarity

    listOfColumns =  [x.strip() for x in params.split(',')]
    #listOfColumns = ['RV']

    clusters = getDataToSim(conn, listOfColumns)

    clusters = clusters.dropna() # drop all the lines with null columns

    originalIDs = clusters.index.values.tolist()

    #clusters = dealingwithNan(clusters)

    if normalize:
        clusters = normalizeDataset(clusters)
    else:
        clusters = np.array(clusters)


    #clusters = multiplyColumn(clusters, 0, 1)

    xSize, ySize = clusters.shape

    #if ySize==1:

    #    clusters = reshapeSizeOneArray(clusters)

    xSize, ySize = clusters.shape

    count = 0

    with open(simPathDir+params+similarityMeasure+'.csv', "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        for cluster in clusters:

            cluster = np.reshape(cluster, (-1, ySize))

            similarityMatrix = getSimilarityMatrixCdist(cluster,  clusters, similarityMeasure)
            for id, sim in zip(originalIDs, similarityMatrix[0]):
                writer.writerow([originalIDs[count]+1, id+1, sim])

                #print originalIDs[count]+1, ",", id+1, "," , sim

            count+=1






