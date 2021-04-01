import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: Meriç İkiz, Uğur Doruk Kırbeyi
GROUP NUMBER: 49
STUDENT ID: 5070767, 5001862
KAGGLE ID: ğ
"""

#####
##
## DATA IMPORT
##
#####
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])



def toRatingMatrix(ratings):
    ratingMatrix = ratings.pivot(index='userID', columns='movieID', values='rating')
    return ratingMatrix

def normalizeRatingForItem(ratingMatrix, meansPerMovie):
    return ratingMatrix.sub(meansPerMovie, axis='columns').fillna(0)

def normalizeUserRatings(ratingMatrix, meansPerUser):
    return ratingMatrix.sub(meansPerUser, axis='index').fillna(0)



def useruserSimilarity(ratingMatrix, spearman):
    if (not spearman):
        return ratingMatrix.T.corr('pearson')
    else:
        return ratingMatrix.T.corr('spearman')

def moviemovieSimilarity(ratingMatrix):
    return ratingMatrix.corr('pearson')



def userAverageRatings(ratingsMatrix): #average per user
    return ratingsMatrix.mean(axis=1, skipna=True)

def movieAverageRatings(ratingsMatrix):
    return ratingsMatrix.mean(axis=0, skipna=True)

def globMean(ratingMatrix):
    return np.nanmean(ratingMatrix.values)



def getMovieBias(movieAverages, globalMean):
    return movieAverages.subtract(globalMean)

def getUserBias(userMeans): #takes a series argument
    globalUserMean = np.mean(userMeans.to_numpy())
    return userMeans.subtract(globalUserMean)





def fillMatrix(ratingMatrix, baselineMatrix):
    cols = ratingMatrix.columns.values
    rows = ratingMatrix.index.values
    allcols = np.arange(1, np.amax(cols)+1)
    allrows = np.arange(1, np.amax(rows)+1)
    colsNotIn = np.invert(np.isin(allcols, cols))
    rowsNotIn = np.invert(np.isin(allrows, rows))
    missingCols = np.sort(allcols[colsNotIn])
    missingRows = np.sort(allrows[rowsNotIn])

    for i in np.arange(len(missingCols)):
        indexColumn = missingCols[i]
        ratingMatrix.insert(loc=int(indexColumn)-1, column=indexColumn, value=np.nan)
    for i in np.arange(len(missingRows)):
        indexRow = missingRows[i]
        array = np.array([indexRow])
        zerosMatrix = pd.DataFrame(np.nan, columns=allcols, index=array)
        ratingMatrix = insert_row(indexRow-1, ratingMatrix, zerosMatrix)
    ratingMatrix[ratingMatrix.isnull()] = baselineMatrix
    return ratingMatrix

def svdDecomposition(filledRatingMatrix):
    ratingsArray = filledRatingMatrix.values
    u, sigma, vt = np.linalg.svd(ratingsArray, full_matrices=False)
    sigmaDiag = np.diag(sigma)

    return sigma, u, sigmaDiag, vt


def dimensionalityReduction(k, fullU, fullSigma, fullVt):
    u = fullU[:, 0:k]
    sigma = fullSigma[0:k, 0:k]
    vt = fullVt[0:k,:]
    return np.matmul(u, np.matmul(sigma, vt))

def decideOnK(sigma, thresholdSVD):
    sum = np.sum(np.square(sigma[1:len(sigma)]))
    currentEnergy = 0
    condition = True
    count = 1
    while condition:
        currentEnergy = currentEnergy + (sigma[count])**2
        ratio = currentEnergy/sum
        if(ratio < thresholdSVD):
            count = count+1
        else:
            count = count+1
            condition = False
    return count

def insert_row(idx, df, df_insert):
    return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ])


def latentFactors(ratingMatrix, baselineMatrix, thresholdSVD):
    filledRatingMatrix = ratingMatrix.copy(deep=True)
    filledRatingMatrix = fillMatrix(filledRatingMatrix, baselineMatrix)
    sigma, u, sigmaDiag, vt = svdDecomposition(filledRatingMatrix)
    k = decideOnK(sigma, thresholdSVD)
    print("length of sigmas is", len(sigma))
    print("svd decomposition dimension k =", k)
    arrayAfterSVD  = dimensionalityReduction(k, u, sigmaDiag, vt)
    return arrayAfterSVD


def getUserMovieBaselineMatrix(ratingMatrix):
    #getting missing cols and rows
    cols = ratingMatrix.columns.values #ndarray
    rows = ratingMatrix.index.values
    allcols = np.arange(1, np.amax(cols)+1)
    allrows = np.arange(1, np.amax(rows)+1)
    colsInBoolean = np.isin(allcols, cols)
    rowsInBoolean = np.isin(allrows, rows)
    colsIn = allcols[colsInBoolean]
    rowsIn = allrows[rowsInBoolean]
    colsNotIn = np.invert(colsInBoolean)
    rowsNotIn = np.invert(rowsInBoolean)
    missingMovies = np.sort(allcols[colsNotIn])
    missingUsers = np.sort(allrows[rowsNotIn])
    meansPerUser = userAverageRatings(ratingMatrix) #this is a series
    movieAverages = movieAverageRatings(ratingMatrix)
    globalMean = globMean(ratingMatrix)
    movieBias = getMovieBias(movieAverages, globalMean)
    userBias = getUserBias(meansPerUser)
    baselineMatrix = pd.DataFrame(0, columns=np.arange(1,len(allcols)+1), index=np.arange(1,len(allrows)+1), dtype=float)
    for user in np.arange(1, len(allrows)+1):
        for movie in np.arange(1, len(allcols)+1):
            userIsMissing = user in missingUsers
            movieIsMissing = movie in missingMovies
            if(movieIsMissing and userIsMissing):
                baselineMatrix.at[user, movie] = globalMean
            elif(userIsMissing):
                baselineMatrix.at[user, movie] = movieAverages[movie]
            elif(movieIsMissing):
                baselineMatrix.at[user, movie] = meansPerUser[user]
            else:
                baselineMatrix.at[user, movie] = globalMean + userBias.at[user] + movieBias.at[movie]
    return baselineMatrix, colsIn, rowsIn, meansPerUser




def userNeighbors(useruserSim, numOfUsers): #parameter to change
    nearestNeigh = np.zeros((numOfUsers, numOfUsers), dtype=int)
    for user, similarities in useruserSim.iterrows(): #similarities is a series
        similarities = similarities.sort_values(axis=0, ascending=False)
        mostSimilarUsersRow = similarities.index.values
        nearestNeigh[(user-1):user, :len(mostSimilarUsersRow)] = np.array(mostSimilarUsersRow).reshape(1, len(mostSimilarUsersRow))
    return nearestNeigh

def itemNeighbours(moviemoviesim, numOfMovies):
    nearestNeigh = np.zeros((numOfMovies, numOfMovies), dtype=int)
    for movie, similarities in moviemoviesim.iterrows(): #similarities is a series
        similarities = similarities.sort_values(axis=0, ascending=False)
        mostSimilarMoviesRow = similarities.index.values
        nearestNeigh[(movie-1):movie, :len(mostSimilarMoviesRow)] = np.array(mostSimilarMoviesRow).reshape(1, len(mostSimilarMoviesRow))
    return nearestNeigh




def kNeighboursUser(x, i, ratingsMatrix, neighboursMatrix, k, useruserSim):
    kNeighbors = [] #empty list
    allneighbors = neighboursMatrix[x-1]
    index = 0
    while (index < k):
        if(index >= np.shape(allneighbors)[0]):
            return kNeighbors
        userJ = allneighbors[index]
        if(userJ == 0):
            return kNeighbors
        ratingOfUserJOnMoviei = ratingsMatrix.at[userJ, i]
        ratingIsNotNan = (not np.isnan(ratingOfUserJOnMoviei))
        usersSimilarityPositive = useruserSim.at[userJ, x] > 0
        if(ratingIsNotNan and usersSimilarityPositive):
            kNeighbors.append(userJ)
        else:
            k = k +1
        index = index +1
    return kNeighbors

def kNeighboursItem(x, i, ratingsMatrix, neighboursMatrix, k, moviemovieSim):
    kNeighbors = [] #empty list
    allneighbors = neighboursMatrix[i-1]
    index = 0
    while (index < k):
        if(index >= np.shape(allneighbors)[0]):
            return kNeighbors
        similarMovie = allneighbors[index]
        if(similarMovie == 0):
            return kNeighbors
        ratingOfUserXOnSimilarMovie = ratingsMatrix.at[x, similarMovie]
        ratingIsNotNan = (not np.isnan(ratingOfUserXOnSimilarMovie))
        movieSimilarityNotZero = moviemovieSim.at[similarMovie, i] > 0
        if(ratingIsNotNan and movieSimilarityNotZero):
            kNeighbors.append(similarMovie)
        else:
            k = k +1
        index = index +1
    return kNeighbors



def predictUserUserForUserMoviePair(x, i, ratingMatrix, k, useruserSimilarity, nearestNeigh, baselineMatrix):
    kNeigh = kNeighboursUser(x, i, ratingMatrix, nearestNeigh, k, useruserSimilarity)
    sumRatings = 0
    sumSimilarity = 0
    if(len(kNeigh) == 0):
        return baselineMatrix.at[x, i]
    else:
        for index in np.arange(len(kNeigh)):
            userJ = kNeigh[index]
            similarityUserXJ = useruserSimilarity.at[userJ, x]
            sumSimilarity = sumSimilarity + similarityUserXJ
            userJRating = ratingMatrix.at[userJ, i]
            sumRatings = sumRatings + (userJRating *similarityUserXJ)
        return sumRatings/sumSimilarity

def predictItemItemForUserMoviePair(x, i, ratingMatrix, k, moviemovieSim, nearestNeigh, baselineMatrix):
    kNeigh = kNeighboursItem(x, i, ratingMatrix, nearestNeigh, k, moviemovieSim)
    sumRatings = 0
    sumSimilarity = 0
    if(len(kNeigh) == 0):
        return baselineMatrix.at[x, i]
    else:
        for index in np.arange(len(kNeigh)):
            movieJ = kNeigh[index]
            similarityMovieIJ = moviemovieSim.at[i, movieJ]
            sumSimilarity = sumSimilarity + similarityMovieIJ
            userRatingOnJ = ratingMatrix.at[x, movieJ]
            sumRatings = sumRatings + (userRatingOnJ *similarityMovieIJ)
        return sumRatings/sumSimilarity


def separatePredictions(userID, movieID,
                        ratingMatrix, baselineMatrix, arrayAfterSVD,
                        neighBourNumberUser, neighBourNumberMovie,
                        moviemovieSim, useruserSim,
                        userNeigh, movieNeigh):
    itemitem = predictItemItemForUserMoviePair(userID, movieID, ratingMatrix, neighBourNumberMovie, moviemovieSim, movieNeigh, baselineMatrix)
    useruser = predictUserUserForUserMoviePair(userID, movieID, ratingMatrix, neighBourNumberUser, useruserSim, userNeigh, baselineMatrix)
    latent = arrayAfterSVD[userID-1, movieID-1]
    baseline = baselineMatrix.at[userID, movieID]
    return itemitem, useruser, latent, baseline


def putTogether(itemitem, useruser, latent, baseline):
    return (itemitem + useruser + latent)/3


def getMatrices(ratings_description, spearmanCorr):
    ratingMatrix = toRatingMatrix(ratings_description)
    baselineMatrix, colsIn, rowsIn, meansPerUser = getUserMovieBaselineMatrix(ratingMatrix)

    userAverage = userAverageRatings(ratingMatrix) #this is a series
    movieAverage = movieAverageRatings(ratingMatrix)

    itemDataCounts = ratingMatrix.count()
    userNormalizedRatings = normalizeUserRatings(ratingMatrix, userAverage)
    itemNormalizedRatings = normalizeRatingForItem(ratingMatrix, movieAverage)

    moviemovie = moviemovieSimilarity(itemNormalizedRatings)
    useruser = useruserSimilarity(userNormalizedRatings, spearmanCorr)

    numOfUsers = np.amax(users_description["userID"].tolist())
    numOfMovies = np.amax(movies_description["movieID"].tolist())

    userNeigh = userNeighbors(useruser, numOfUsers=numOfUsers)
    itemNeigh = itemNeighbours(moviemovie, numOfMovies)

    return ratingMatrix, baselineMatrix, userAverage, movieAverage, moviemovie, useruser, userNeigh, itemNeigh


def predictIndividual(userID, movieID, baselineMatrix, ratingMatrix, arrayAfterSVD,
                      moviemovieSim, useruserSim, userNeigh, movieNeigh, parameters):
    neighBourNumberUser = parameters[0]
    neighBourNumberMovie = parameters[1]

    movieExists = movieID in ratingMatrix
    userExists = userID in ratingMatrix.index
    if((not userExists) or (not movieExists)):
        rating = baselineMatrix.at[userID, movieID]
    else:
        itemitem, useruser, latent, baseline = separatePredictions(userID, movieID,
                                                               ratingMatrix, baselineMatrix, arrayAfterSVD,
                                                               neighBourNumberUser, neighBourNumberMovie,
                                                               moviemovieSim, useruserSim,
                                                               userNeigh, movieNeigh)
        rating = putTogether(itemitem, useruser, latent, baseline)
    return rating


def predict(ratings_description, predictions_description, parameters):
    spearmanCorr = False
    ratingMatrix, baselineMatrix, userAverage, movieAverage, moviemovie, useruser, userNeigh, itemNeigh = getMatrices(ratings_description, spearmanCorr)
    thresholdSVD = parameters[2]
    arrayAfterSVD = latentFactors(ratingMatrix, baselineMatrix, thresholdSVD)

    predictionsNumber = predictions_description.shape[0]
    predictedRatingsList = list()
    for i in np.arange(0, predictionsNumber):
        userID = predictions_description.iloc[i].get(0)
        movieID = predictions_description.iloc[i].get(1)
        rating = predictIndividual(userID, movieID, baselineMatrix, ratingMatrix, arrayAfterSVD,
                      moviemovie, useruser, userNeigh, itemNeigh, parameters)
        predictedRatingsList.append(rating)

    return list(enumerate(predictedRatingsList, 1)) #tupled list




#####
##
## SAVE RESULTS
##
#####
parameters = np.zeros((3,1)) #neighBourNumberUser, neighBourNumberMovie, thresholdSVD
parameters[0] = 40
parameters[1] = 20
parameters[2] = 0.3

predictions = predict(ratings_description, predictions_description, parameters)


with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)

    #Writes it dowmn
    submission_writer.write(predictions)
