import numpy as np
import pandas as pd
from src.recommenderModel.functions.modelEvaluationMethods import MRR3, MRR4

'''This function applies the backtesting logic. Comparing the back-dated data to the current data,
 flagging if members have new items and if members have a new type of item. Filtering on members with new item types, 
 adding their item recommendations and scoring using specified evaluation method (comparing real new items vs item recommendations)'''

def backtesting(backDatedData,upToDateData,trainedDataIds,idColumnName,trainingDataWithRecommendations, evaluationMethod='MRR4'):
    #a subset of backDatedData was used to train the model and recommendations generated, this data with the recommendations is trainingDataWithRecommendations
    
    backDatedDataSample = pd.merge(backDatedData,trainedDataIds,on=idColumnName,how='inner') #selecting records that were trained on using ids of the training set
    upToDateDataSample = pd.merge(upToDateData,trainedDataIds,on=idColumnName,how='inner') #getting current info for the training set using ids of the training set
    joinedData = pd.merge(backDatedDataSample,upToDateDataSample[[idColumnName,'no. unique items','set of items','item_count']],
                    on = idColumnName,
                    how = 'inner', suffixes=('_bd', '_c')) # bd for back-dated data, c for current/up-to-date data

    #flag whether id has new item and flag for if id has new type of item (bc id can take out new item of the same item type they already have)
    joinedData['new_item_flag'] = (joinedData['item_count_bd'] < joinedData['item_count_c']).astype(int)
    joinedData['new_item_type_flag'] = (joinedData['no. unique items_bd'] < joinedData['no. unique items_c']).astype(int)
    #getting the name of the new product
    joinedData['newestItem'] = [c-bd if f == 1 else 0 
                              for bd,c,f in zip(joinedData['set of items_bd'],joinedData['set of items_c'],joinedData['new_item_type_flag'])]

    newItemData = joinedData[joinedData['newestItem'] != 0] #dataframe of users with new item types
    del joinedData
    newItemData = pd.merge(newItemData,trainingDataWithRecommendations[[idColumnName,'no.1 most recommended item','no.1 item recommendation score','no.2 most recommended item','no.2 item recommendation score','no.3 most recommended item','no.3 item recommendation score','no.4 most recommended item','no.4 item recommendation score','no.5 most recommended item','no.5 item recommendation score','no.6 most recommended item','no.6 item recommendation score']],
                      on = idColumnName, 
                      how='inner') #include recommendations
    
    #scoring
    if evaluationMethod == 'MRR3':
        newItemData = MRR3(newItemData,'newestItem')
        modelScoreColumn = 'MRR_3_score'
    elif evaluationMethod == 'MRR4':
        newItemData = MRR4(newItemData,'newestItem')
        modelScoreColumn = 'MRR_4_score'
    return newItemData,modelScoreColumn
'''returning dataframe with training data, top n recommendations, item info and recommendation scores'''