from src.recommenderModel.functions import dataPreparation, backTesting,gradientDescent, kFolds, modelEvaluationMethods, recommendationScore
from datetime import datetime
import numpy as np

class nonMetricBasedRecommender():
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





        




