import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import normalize

''' 
Cosine similairty function using matrices to find the cosine similarity between all id pairs based on numerical features.
 weighting vector is used to weight the features.'''

def weighted_cosine_similarity(dataMatrix,norm, weightingVector): #dataMatrix is all the data in matrix form, 
    # norm is the type of norm used when normalising the data,
    # weightingVector is a vector of weights for each variable that's data is in the dataMatrix

    #normalise the matrix
    dataMatrix = normalize(dataMatrix, axis=1, norm = norm)
    
    #matrix multiplication of matrix nd it's transpose
    sim_matrix = np.matmul(weightingVector*dataMatrix,dataMatrix.T)
    del dataMatrix
    
    #make diagonals zero
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix


'''Jaccard similairty function to determine the jaccard similarity score between all id pairs using categorical features.
 Weighting vector is used to weight the features.'''

def weighted_jaccard_similarity(dataMatrix, weightingVector):
     #dataMatrix is all the data in matrix form, 
    # weightingVector is a vector of weights for each variable that's data is in the dataMatrix
     
     # Ensure the feature matrix is a TensorFlow tensor
     dataMatrix = tf.convert_to_tensor(dataMatrix, dtype=tf.float32)
     # Compute the intersection (dot product)
     intersection = tf.matmul(weightingVector*dataMatrix, dataMatrix, transpose_b=True)
     # Compute the sum of 1s per row
     row_sums = tf.reduce_sum(weightingVector*dataMatrix, axis=1)
     # Compute the union
     union = row_sums[:, tf.newaxis] + row_sums - intersection
     # Avoid division by zero by setting union to at least 1
     union = tf.clip_by_value(union,clip_value_min=1.0, clip_value_max=tf.reduce_max(union))
     # Compute the Jaccard similarity matrix
     jaccard_sim_matrix = intersection / union
     del dataMatrix,intersection,union,row_sums
     
     # Set diagonal elements to 0
     jaccard_sim_matrix = jaccard_sim_matrix - tf.linalg.diag(tf.linalg.diag_part(jaccard_sim_matrix))
     
     return jaccard_sim_matrix


'''Function to combine the cosine and jaccard similarities using some weighting (num:cat for numerical and categorical features)'''

def similarity_matrix(df,norm,categoricalColumns,cosineWeights,jaccardWeights): #dataframe with cat and num features, norm used for cosine, number of categorical columns, cosine weighting, jaccard weighting
    
    sim_matrix =0.5*weighted_cosine_similarity(df.iloc[:,categoricalColumns:].values,norm,cosineWeights) #similarity based on numerical info
    sim_matrix = sim_matrix + 0.5*weighted_jaccard_similarity(df.iloc[:,0:categoricalColumns],jaccardWeights) #add similarity based on categorical info
    
    return sim_matrix