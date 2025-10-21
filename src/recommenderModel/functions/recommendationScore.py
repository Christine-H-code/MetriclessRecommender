import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.recommenderModel.functions.similarityScores import similarity_matrix

'''Calculate the similarity matrix then the recommendation scores (dot product between the similarity matrix and the item indicators), 
normalizing these scores using the Min-Max normalization method '''

def RecommendationScores(nRecommendations, cosineWeights, jaccardWeights,recommendItemColumns,categoricalColumns, aggregated_df,hotOneEncodingPrefix=None):


    similarityMatrix = similarity_matrix(aggregated_df.copy(),'l2',categoricalColumns,cosineWeights,jaccardWeights)
    itemIndicator_df = aggregated_df[recommendItemColumns]
    itemIndicator_inverse_df = pd.DataFrame(np.logical_xor(itemIndicator_df.values,1).astype(int),columns=itemIndicator_df.columns, index=itemIndicator_df.index)
    scores = np.dot(similarityMatrix,itemIndicator_df)#recommendation score, sum of similarities for items users have (dot product)
    scaler = MinMaxScaler()
    scoresScaledAll = scaler.fit(scores)
    scoresScaledAll = scaler.transform(scores) #apply normalization (MinMax)
    scoresScaledFiltered = scoresScaledAll*itemIndicator_inverse_df #make score zero for items user already has

    scores_df = pd.DataFrame(scoresScaledFiltered,columns=itemIndicator_df.columns, index=itemIndicator_df.index)
    newColumns = {}
    del itemIndicator_df,itemIndicator_inverse_df,scores,scoresScaledFiltered,scoresScaledAll

    #adding recommendations to the aggregated dataframe
    for i in range(1, nRecommendations+1, 1):
        newColumns[f"no.{i} most recommended item"] = []
        newColumns[f"no.{i} item recommendation score"] = []
    for n in aggregated_df.index:

        sortedScores = scores_df.loc[n].sort_values()
        recommendationScores = np.array(sortedScores.values[-nRecommendations:])
        prod = np.array(sortedScores.index[-nRecommendations:])
        for i in range(nRecommendations-1,-1,-1):
            j = nRecommendations-i
            newColumns[f"no.{j} most recommended item"].append(prod[i])
            newColumns[f"no.{j} item recommendation score"].append(recommendationScores[i])

    recommendations_df = pd.DataFrame(aggregated_df.index)
    recommendations_df = recommendations_df.assign(**newColumns)

#Removing the prefix from hot-one encoding
    if hotOneEncodingPrefix is not None:
        for r in range(1,nRecommendations+1,1):
            recommendations_df[f'no.{r} most recommended item'] = recommendations_df[f'no.{r} most recommended item'].str.replace(hotOneEncodingPrefix, '') #cleaning up

    return recommendations_df