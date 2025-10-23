from src.recommenderModel.functions import dataPreparation, backTesting,gradientDescent, kFolds, modelEvaluationMethods, recommendationScore
from datetime import datetime
import numpy as np

class nonMetricBasedRecommender_v1():
    def __init__(self, data, idColumnName: str, itemColumnName: str,kFolds: int = 2, gradientStepSize: float = 0.01, categiricalColumns: list = None, numericalColumns: list = None, backDatingDate: dict = {"dateColumnName":None, "cutOffDate":None}, backDatedData = None,fold_size: int = 10000):
        
        nRecommendations = 6 
        self.originalData = data.copy()
        self.idColumnName = idColumnName
        self.itemColumnName = itemColumnName
        self.kFolds = kFolds
        self.categiricalColumns = categiricalColumns
        self.numericalColumns = numericalColumns
        self.aggUpToDateData = dataPreparation.data_preparation(data.copy(),idColumnName,itemColumnName)[2]

        if backDatedData is not None:
            self.backDatedData = backDatedData.copy()

        elif backDatedData is None and backDatingDate["dateColumnName"] is not None and backDatingDate["cutOffDate"] is not None:
            format_string = "%Y-%m-%d %H:%M:%S"
            cutOffDate = datetime.strptime(backDatingDate["cutOffDate"], format_string)
            self.backDatedData = data[data[backDatingDate["dateColumnName"]] <= cutOffDate].copy()

        else:
            self.backDatedData = None
        
        if self.backDatedData is None:
            self.reccomendationData = "Not able to backdtest for evaluating model, please provide backdated data or provide backdating parameters"
        else:
            itemColumns,nCategoricalColumns,self.aggBackDatedData = dataPreparation.data_preparation(self.backDatedData.copy(),idColumnName,itemColumnName)
            foldSize = min(10000,self.backDatedData.shape[0],fold_size)
            cosine_weightingsPerFold,jaccard_weightingsPerFold,optimalScoresPerFold = kFolds.k_folds_optimization( self.backDatedData, kFolds, self.aggBackDatedData, self.aggUpToDateData, nRecommendations, gradientStep=gradientStepSize, foldSize=foldSize, fold_sample_size=min(1000,foldSize/2))
            self.optimalJaccardWeightings = jaccard_weightingsPerFold.mean(axis=0)
            self.optimalCosineWeightings = cosine_weightingsPerFold.mean(axis=0)
            del cosine_weightingsPerFold, jaccard_weightingsPerFold, optimalScoresPerFold

            scoresPerFold = kFolds.k_folds( self.backDatedData, kFolds, self.aggBackDatedData, self.aggUpToDateData, nRecommendations, self.optimalCosineWeightings, self.optimalJaccardWeightings, foldSize=foldSize)
            self.modelScore = np.mean(scoresPerFold)

            del scoresPerFold

            self.recommendationData = recommendationScore.RecommendationScores(nRecommendations,self.optimalCosineWeightings, self.optimalJaccardWeightings,itemColumns,nCategoricalColumns,self.aggUpToDateData,itemColumnName )


class nonMetricBasedRecommender_v2():
    def __init__(self, data, idColumnName: str, itemColumnName: str,kFolds: int = 2, gradientStepSize: float = 0.01, categiricalColumns: list = None, numericalColumns: list = None, fold_size: int = 10000,tetsingMethod: str = "remove-one",evaluationMethod: str = "MRR4", backDatedData = None, backDatingDate: dict = {"dateColumnName":None, "cutOffDate":None}):
        #!!!!!!!!!!!!!!!
        #undo hardcoding and make it specifiable and have it be the min of 6 and specified value
        self.nRecommendations = 6
        #!!!!!!!!!!!!!!!!!!
        self.originalData = data.copy()
        self.idColumnName = idColumnName
        self.itemColumnName = itemColumnName
        self.kFolds = kFolds
        self.categiricalColumns = categiricalColumns
        self.numericalColumns = numericalColumns
        self.gradientStepSize = gradientStepSize
        self.aggTestData = dataPreparation.data_preparation(data.copy(),idColumnName,itemColumnName)[2]
        self.evaluationMethod = evaluationMethod

        if tetsingMethod == "back-testing":
            if backDatedData is not None:
                self.trainingData = backDatedData.copy()

            elif backDatedData is None and backDatingDate["dateColumnName"] is not None and backDatingDate["cutOffDate"] is not None:
                format_string = "%Y-%m-%d %H:%M:%S"
                cutOffDate = datetime.strptime(backDatingDate["cutOffDate"], format_string)
                self.trainingData = data[data[backDatingDate["dateColumnName"]] <= cutOffDate].copy()

            else:
                #same as for if evaluation method != "back-testing"
                pass

        else:
            #code to be written... remove items from ids to mimic backdated data to mock backtesting
            pass

        self.itemColumns,self.nCategoricalColumns,self.aggTrainingData = dataPreparation.data_preparation(self.trainingData.copy(),idColumnName,itemColumnName)
        self.foldSize = min(10000,self.aggTrainingData.shape[0],fold_size)
        
        
    def fit(self):
        cosine_weightingsPerFold,jaccard_weightingsPerFold,optimalScoresPerFold = kFolds.k_folds_optimization( self.trainingData, self.kFolds, self.aggTrainingData, self.aggTestData, self.nRecommendations, gradientStep=self.gradientStepSize, foldSize=self.foldSize, fold_sample_size=min(1000,self.foldSize/2),evaluationMethod=self.evaluationMethod)
        self.optimalJaccardWeightings = jaccard_weightingsPerFold.mean(axis=0)
        self.optimalCosineWeightings = cosine_weightingsPerFold.mean(axis=0)
    
    def feature_importance(self):
        #!!!!!!!!!!!!!!!!!!!!!!!!
        #add the feature names to the weights
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return self.optimalCosineWeightings, self.optimalJaccardWeightings
    
    def score(self):
        MRR3scoresPerFold = kFolds.k_folds( self.trainingData, self.kFolds, self.aggTrainingData, self.aggTestData, self.nRecommendations, self.optimalCosineWeightings, self.optimalJaccardWeightings, foldSize=self.foldSize,evaluationMethod="MRR3")
        MRR3modelScore = np.mean(MRR3scoresPerFold)
        MRR4scoresPerFold = kFolds.k_folds( self.trainingData, self.kFolds, self.aggTrainingData, self.aggTestData, self.nRecommendations, self.optimalCosineWeightings, self.optimalJaccardWeightings, foldSize=self.foldSize,evaluationMethod="MRR4")
        MRR4modelScore = np.mean(MRR4scoresPerFold)
        return {'MRR_3 score':MRR3modelScore, 'MRR_4 score': MRR4modelScore}
    
    def recommend(self):
        self.recommendationData = recommendationScore.RecommendationScores(self.nRecommendations,self.optimalCosineWeightings, self.optimalJaccardWeightings,self.itemColumns,self.nCategoricalColumns,self.aggTestData,self.itemColumnName )
        return self.recommendationData
    
    def score_progression(self):
        #!!!!!!!!!!!!!!!!!
        #edit gradient descent code to return scores per iteration to use here to plot, this will be returned when running fit(), kfold_optimization
        #!!!!!!!!!!!!!!!!!!
        pass

    def feature_importance_plot(self):
        #plots out the feature importance
        pass




        


        




