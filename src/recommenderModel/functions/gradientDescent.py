
import numpy as np
import pandas as pd
from src.recommenderModel.functions.recommendationScore import RecommendationScores
from src.recommenderModel.functions.backTesting import backtesting




'''Calculated the gradient of the loss graph for step in each weighting to find change of loss for gradient descent update formula'''
def gradient(cosineWeights,jaccardWeights,gradientStep,trainingDataSample, itemColumns, categoricalColumns, idColumnName,backDatedData,upToDateData,trainedDataIds,evaluationMethod):
  loss_step = []
  for i in range(0,len(cosineWeights)):
      cw_step = list(cosineWeights)
      cw_step[i] += gradientStep

      trainedData = RecommendationScores(6, cw_step, jaccardWeights,itemColumns,categoricalColumns,trainingDataSample.copy())
      backtestedData,modelScoreColumn = backtesting(backDatedData,upToDateData,trainedDataIds,idColumnName,trainedData,evaluationMethod)
      modelScoreStep = np.mean(backtestedData[modelScoreColumn])
      del backtestedData

      loss_step.append(1-modelScoreStep) #1-score to mimic log loss where 0 means perfect performing model
      
  for i in range(0,len(jaccardWeights)):
      jw_step = list(jaccardWeights)
      jw_step[i] += gradientStep

      trainedData = RecommendationScores(6, cosineWeights, jw_step,itemColumns,categoricalColumns,trainingDataSample.copy())
      backtestedData,modelScoreColumn = backtesting(backDatedData,upToDateData,trainedDataIds,idColumnName,trainedData,evaluationMethod)
      modelScoreStep = np.mean(backtestedData[modelScoreColumn])
      del backtestedData
      
      
      loss_step.append(1-modelScoreStep)
      

  return np.array(loss_step)

'''returns the change in loss accross all weightings'''



'''update for weighting with gradient descent method using the change in "loss" from the gradient function'''
def gradient_descent(cosineWeights,jaccardWeights,itemColumns,categoricalColumns,trainingDataSample,backDatedData,upToDateData,idColumnName,gradientStep,evaluationMethod='MRR4'):
    trainedData = RecommendationScores(6, cosineWeights, jaccardWeights,itemColumns,categoricalColumns,trainingDataSample.copy())
    trainedDataIds = trainedData[idColumnName].drop_duplicates() #trainedDataIds that have recommendations, training trainedDataIds

    backtestedData,modelScoreColumn = backtesting(backDatedData,upToDateData,trainedDataIds,idColumnName,trainedData,evaluationMethod) #backtesting on the current fold

    mod_score = np.mean(backtestedData[modelScoreColumn])
    del backtestedData
    
    loss_step = gradient(cosineWeights,jaccardWeights,gradientStep,trainingDataSample, itemColumns, categoricalColumns, idColumnName,backDatedData,upToDateData,trainedDataIds) #loss for each weighting
    d_loss = loss_step - np.full_like(loss_step, 1- mod_score)#differnce in loss

    cw_next = cosineWeights - d_loss[0:len(cosineWeights)]
    jw_next = jaccardWeights - d_loss[len(cosineWeights):]
    
    return cw_next, jw_next, np.linalg.norm(d_loss)

'''return the updated weigtings and the loss difference'''