import numpy as np
import pandas as pd
from src.recommenderModel.functions.recommendationScore import RecommendationScores
from src.recommenderModel.functions.backTesting import backtesting
from src.recommenderModel.functions.gradientDescent import gradient_descent
from src.recommenderModel.functions.dataPreparation import data_preparation



'''runs k folds of training sets (some sample of backdated data) through recommender, and backtests each. 
If best weigthings are known and no optimization is necessary, use this function instead of k_folds_optimization.'''
def k_folds(data,k,backDatedData,upToDateData,nRecommendations,cosineWeights,jaccardWeights,foldSize = 1000,evaluationMethod='MRR4'):

    foldScores = []

    #training data (subset of backdated data)
    itemColumns,nCategoricalColumns, trainingData,idColumnName,itemColumnName = data_preparation(data.copy()) 

    trainingData = trainingData.drop(columns = ['item_count','no. unique items','set of items']).set_index(idColumnName)
   
    for i in range(k):
        
        trainedData = RecommendationScores(nRecommendations, cosineWeights, jaccardWeights,itemColumns,nCategoricalColumns,trainingData.copy().sample(foldSize),itemColumnName ) #recommender function to get recommendations for the fold sample
        
        trainedDataIds = trainedData[idColumnName].drop_duplicates() #ids that have recommendations, training ids

        backtestedData,modelScoreColumn = backtesting(backDatedData,upToDateData,trainedDataIds,idColumnName,trainedData.copy(),evaluationMethod) #backtesting on the current fold

        foldScores.append(np.mean(backtestedData[modelScoreColumn]))
        
    return foldScores

'''returns the ave model score for all folds'''



'''runs k folds of training sets (some sample of backdated data) through recommender, and backtests each'''
def k_folds_optimization(data,k,backDatedData,upToDateData,nRecommendations,gradientStep,foldSize = 1000,fold_sample_size = 1000,evaluationMethod='MRR4'):


    foldScores = [] #list of scores for each fold
    cosine_weightings = [] #list of cosine weightings used for each fold
    jaccard_weightings = [] #list of jaccard weightings used for each fold

    #training data (subset of backdated data)
    itemColumns,nCategoricalColumns, trainingData,idColumnName,itemColumnName = data_preparation(data.copy()) 

    trainingData = trainingData.drop(columns = ['item_count','no. unique items','set of items']).set_index(idColumnName)
    
    for i in range(k):
        e = 0.1 #initializing error value for gradient descent
        t = 70 #to avoid infinite loop

        trainingDataSample = trainingData.copy().sample(fold_sample_size) #sample of the fold of data used for optimization

        cosineWeights = (1)*np.array([1] *(trainingDataSample.shape[1] -nCategoricalColumns)) 
        jaccardWeights = (1)*np.array([1] *(nCategoricalColumns)) #include item info

        #--------------------optimizations of weightings------------------
        #process of finding optimal weightings on fold sample

        while e > 0.005 and t>0: #as e tend to zero the change in the "loss" value tends to zero
            cosineWeights,jaccardWeights,e = gradient_descent(cosineWeights,jaccardWeights,itemColumns,nCategoricalColumns,trainingDataSample,backDatedData,upToDateData,idColumnName,gradientStep,evaluationMethod=evaluationMethod,itemColumnName=itemColumnName) 
            
            t = t-1 

        #------------------------------------------------------------------------
        #found optimal weigthings applied to the full fold for final fold scoring
        trainedData = RecommendationScores(nRecommendations, cosineWeights, jaccardWeights,itemColumns,nCategoricalColumns,trainingData.copy().sample(foldSize),itemColumnName)
        trainedDataIds = trainedData[idColumnName].drop_duplicates() #trainedDataIds that have recommendations, training trainedDataIds

        backtestedData,modelScoreColumn = backtesting(backDatedData,upToDateData,trainedDataIds,idColumnName,trainedData.copy(),evaluationMethod)

        foldScores.append(np.mean(backtestedData[modelScoreColumn])) #average model score for the fold
        cosine_weightings.append(cosineWeights)
        jaccard_weightings.append(jaccardWeights)
    return cosine_weightings,jaccard_weightings,foldScores

'''returns the optimal weightings for all folds and the ave model score for all folds'''
