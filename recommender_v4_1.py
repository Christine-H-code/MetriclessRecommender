import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf 
import math as m
import gc
import random



# more libraries required
from sklearn.preprocessing import OneHotEncoder, normalize, MinMaxScaler

def prep(df):
    pol_col = df.columns[-1] #name of policy column
    id_col = df.columns[0] #name of id column
     # Count the number of policies per user
    df['pol_count'] = 1
    temp_df = df.groupby(id_col)['pol_count'].sum().reset_index()
    # Merge policy count back into the original DataFrame
    df = df.drop(columns={'pol_count'})
    df = pd.merge(df, temp_df, 
                         on=id_col, 
                         how='left')
    # Create a set of unique policies per user
    temp_df = pd.DataFrame(df.groupby(id_col)[pol_col].apply(set))
    temp_df = temp_df.rename(columns={pol_col: 'set of policies'})  #column containing a set of all policy types client has

    df = pd.merge(df, temp_df,on = id_col, how = 'left')

    # Count unique policy types per user
    df['no unique products'] = df['set of policies'].apply(lambda x: len(x) if isinstance(x, set) else 0)

    # Process numerical features
    num_df = df.iloc[:,1:].select_dtypes(include=['float64','int64']).copy() #creating numerical features df
    num_df.insert(0,id_col,df[id_col])
    num_df.iloc[:,1:] = num_df.iloc[:,1:].fillna(0.0)

    def custom_agg(series): #if the numerical value is the same accross all rows per client don't take sum, take value (eg, how many policies), otherwise take sum (eg. premium for policy)
        return series.iloc[0] if len(series.unique()) == 1 else series.sum()
    # Group by 'id' and apply the aggregation
    num_df = num_df.groupby(id_col).agg({col: custom_agg for col in num_df.columns[1:]}).reset_index()

    #creating categorical feature df
    cat_df = df.iloc[:,1:-2].select_dtypes(include=['object','bool','category']).copy()
    cat_df.insert(0,id_col,df[id_col])
    cat_df = pd.get_dummies(cat_df.iloc[:,1:],columns = cat_df.iloc[:,1:].columns, dtype=int)
    cat_df.insert(0,id_col,df[id_col])
    cat_df = cat_df.groupby(id_col).sum()
    cat_df = cat_df[cat_df.columns].map(lambda x: 1 if x > 0 else 0)

    df = df[[id_col, 'set of policies']]
    # Convert set columns to sorted tuples so they're hashable and comparable
    df['set of policies'] = df['set of policies'].apply(
    lambda x: tuple(sorted(x)) if isinstance(x, set) else tuple())
    df = df.drop_duplicates()
    df['set of policies'] = df['set of policies'].apply(set)

    #merging categorical and numerical aggregated data
    agg_df = pd.merge(cat_df, num_df,on = id_col, how = 'inner') #aggregated dataframe
    agg_df = pd.merge(agg_df, df,on = id_col, how = 'inner')
    cat_col = len(cat_df.columns) #how many categorical columns there are

    pol_columns = [col for col in agg_df.columns if col.startswith(pol_col)] #list of all policy columns

    return pol_columns,cat_col, agg_df,id_col
'''returns policy columns names, number of categorical columns, the aggregated dataframe and the id column name'''

#------------------------------------------------------------------------------------------------------------------------------------------------------------------


''' 
Cosine similairty function using matrices to find the cosine similarity between all id pairs based on numerical features. weighting vector is used to weight the features.'''
def weighted_cosine_similarity(A,n, w): #w is the weighting vector
    #normalise the matrix
    A = normalize(A, axis=1, norm = n)
    
    #matrix multiplication of matrix nd it's transpose
    sim_matrix = np.matmul(w*A,A.T)
    del A
    
    #make diagonals zero
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Jaccard similairty function to determine the jaccard similarity score between all id pairs using categorical features. Weighting vector is used to weight the features.'''

def weighted_jaccard_similarity(feature_matrix, w):  #w is the weighting vector
     
     # Ensure the feature matrix is a TensorFlow tensor
     feature_matrix = tf.convert_to_tensor(feature_matrix, dtype=tf.float32)
     # Compute the intersection (dot product)
     intersection = tf.matmul(w*feature_matrix, feature_matrix, transpose_b=True)
     # Compute the sum of 1s per row
     row_sums = tf.reduce_sum(w*feature_matrix, axis=1)
     # Compute the union
     union = row_sums[:, tf.newaxis] + row_sums - intersection
     # Avoid division by zero by setting union to at least 1
     union = tf.clip_by_value(union,clip_value_min=1.0, clip_value_max=tf.reduce_max(union))
     # Compute the Jaccard similarity matrix
     jaccard_sim_matrix = intersection / union
     del feature_matrix,intersection,union,row_sums
     
     # Set diagonal elements to 0
     jaccard_sim_matrix = jaccard_sim_matrix - tf.linalg.diag(tf.linalg.diag_part(jaccard_sim_matrix))
     
     return jaccard_sim_matrix

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Function to combine the cosine and jaccard similarities using some weighting (num:cat for numerical and categorical features)'''

def similarity_matrix(df,norm,cat_col,cw,jw): #dataframe with cat and num features, norm used for cosine, number of categorical columns, cosine weighting, jaccard weighting
    
    sim_matrix =0.5*weighted_cosine_similarity(df.iloc[:,cat_col:].values,norm,cw) #similarity based on numerical info
    sim_matrix = sim_matrix + 0.5*weighted_jaccard_similarity(df.iloc[:,0:cat_col],jw) #add similarity based on categorical info
    
    return sim_matrix


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''Calculate the similarity matrix then the recommendation scores (dot product between the similarity matrix and the policy indicators), normalizing these scores using the Min-Max normalization method '''
def recommender(recs, cw, jw,pol_columns,cat_col, agg_df):


    sim_matrix = similarity_matrix(agg_df.copy(),'l2',cat_col,cw,jw)
    policy_ind_df = agg_df[pol_columns]
    policy_ind_inverse_df = pd.DataFrame(np.logical_xor(policy_ind_df.values,1).astype(int),columns=policy_ind_df.columns, index=policy_ind_df.index)
    scores = np.dot(sim_matrix,policy_ind_df)#recommendation score, sum of similarities for products users have (dot product)
    scaler = MinMaxScaler()
    scores_scaled_r = scaler.fit(scores)
    scores_scaled_r = scaler.transform(scores) #apply normalization (MinMax)
    scores_scaled = scores_scaled_r*policy_ind_inverse_df #make score zero for products user already has
    scores_df = pd.DataFrame(scores_scaled,columns=policy_ind_df.columns, index=policy_ind_df.index)
    new_col = {}
    del policy_ind_df,policy_ind_inverse_df,scores,scores_scaled,scores_scaled_r

    #adding recommendations to the aggregated dataframe
    for i in range(1, recs+1, 1):
        new_col[f"no.{i} most rec product"] = []
        new_col[f"no.{i} product score"] = []
    for n in agg_df.index:

        sorted = scores_df.loc[n].sort_values()
        rec_sc = np.array(sorted.values[-recs:])
        prod = np.array(sorted.index[-recs:])
        for i in range(recs-1,-1,-1):
            j = recs-i
            new_col[f"no.{j} most rec product"].append(prod[i])
            new_col[f"no.{j} product score"].append(rec_sc[i])

    rec_df = pd.DataFrame(agg_df.index)
    rec_df = rec_df.assign(**new_col)


    for r in range(1,recs+1,1):
        rec_df[f'no.{r} most rec product'] = rec_df[f'no.{r} most rec product'].str.replace('pol_type_desc_', '') #cleaning up

    return rec_df

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Function to calculate the mean reciprical rank score based on the recommendations and the new product taken out.'''

def MRR(df,pol_col): #scoring based on where new product sits in regards to recommendations, mean reciprical rank_3 and 4
 
    conditions = [df.apply(lambda row: len({row['no.1 most rec product']}.intersection(row[pol_col])) ,axis=1) == 1,
                  df.apply(lambda row: len({row['no.2 most rec product']}.intersection(row[pol_col])) ,axis=1) == 1, 
                  df.apply(lambda row: len({row['no.3 most rec product']}.intersection(row[pol_col])) ,axis=1) ==1]
    scores = [1, 0.66, 0.33]

    df['MRR_3_score'] = np.select(conditions, scores, default=0)

    df['MRR_4_score'] = [
    1 if f_rec in s else 0.75 if s_rec in s else 0.5 if t_rec in s else 0.25 if fr_rec in s else 0
    for f_rec,s_rec,t_rec,fr_rec,s in zip(df['no.1 most rec product'],df['no.2 most rec product'],df['no.3 most rec product'],df['no.4 most rec product'],df['newest_product'])]

    return df

'''df is the dataframe given with MRR scores appended'''

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''This function applies the backtesting logic. Comparing the back-dated data to the current data,
 flagging if members have new policies and if members have a new product. Filtering on members with new products, 
 adding their product recommendations and scoring using Mean Reciprocal Rank (comparing real new products taken out vs recommendations)'''

def backtesting(Dx,Dc,pids,id_col,Rx):
    
    Dx_s = pd.merge(Dx,pids,on=id_col,how='inner') #selecting records that were trained on
    Dc_s = pd.merge(Dc,pids,on=id_col,how='inner') #getting current info for the training pids
    df_j = pd.merge(Dx_s,Dc_s[[id_col,'no unique products','set of policies','pol_count']],
                    on = id_col,
                    how = 'inner', suffixes=('_x', '_c'))

    #flag whether client has new policy and flag for if client has new product (bc client can take ut new policy of product they already have)
    df_j['new_policy_flag'] = (df_j['pol_count_x'] < df_j['pol_count_c']).astype(int)
    df_j['new_product_flag'] = (df_j['no unique products_x'] < df_j['no unique products_c']).astype(int)
    #getting the name of the new product
    df_j['newest_product'] = [c-x if f == 1 else 0 
                              for x,c,f in zip(df_j['set of policies_x'],df_j['set of policies_c'],df_j['new_product_flag'])]

    df_new = df_j[df_j['newest_product'] != 0] #dataframe of users with new products
    del df_j
    df_new = pd.merge(df_new,Rx[[id_col,'no.1 most rec product','no.1 product score','no.2 most rec product','no.2 product score','no.3 most rec product','no.3 product score','no.4 most rec product','no.4 product score','no.5 most rec product','no.5 product score','no.6 most rec product','no.6 product score']],
                      on = id_col, 
                      how='inner') #include recommendations
    
    #scoring
    df_new = MRR(df_new,'newest_product')

    return df_new
'''returning dataframe with training data, recommendations, policy info and recommendation scores'''

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''runs k folds of training sets (some sample of backdated data) through recommender, and backtests each. If best weigthings are known and no optimization is necessary, se this function instead of k_folds_optimization.'''
def k_folds(df_x,k,Dx,Dc,recs,cw,jw,fold_size = 10000):

    scores = []

    #training data (subset of backdated data)
    pol_columns_r,cat_col_r, Rx,id_col = prep(df_x.copy().drop(columns = ['businessentityname', 'trn_grp_desc', 'prod_house_desc',
           'prod_tbl_grp_desc', 'psm_opt_desc',  'sky_pol_type_desc', 'product_desc', 'prod_tbl_cd', 'pa_sts_desc','business_only_indicator', 
           'death_benefit_indicator','pps_full_flag'])) 
    #pol_columns_r,cat_col_r, Rx,id_col = prep(df_x.copy())
    #columns dropped because of previous model's target leakage, but model performs better without these columns...
    #Maybe investigate effect of keeping/dropping certain columns
    Rx = Rx.drop(columns = ['pol_count','no unique products','set of policies']).set_index(id_col)
   
    for i in range(k):
        
        Rx_f = recommender(recs, cw, jw,pol_columns_r,cat_col_r,Rx.copy().sample(fold_size) ) #recommender function to get recommendations for the fold sample
        
        pids = Rx_f[id_col].drop_duplicates() #pids that have recommendations, training pids

        df_new = backtesting(Dx,Dc,pids,id_col,Rx_f.copy()) #backtesting on the current fold

        scores.append(np.mean(df_new['MRR_4_score']))
        
    return df_new,scores

'''returns the latest fold's dataframe with recs and the ave MRR_4 model score for all folds'''

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''Calculated the gradient of the loss graph for step in each weighting to find change of loss for gradient descent update formula'''
def gradient(cw,jw,step,Rx, pol_columns, cat_col, id_col,Dx,Dc,pids):
  loss_step = []
  for i in range(0,len(cw)):
      cw_step = list(cw)
      cw_step[i] += step

      recs_df = recommender(6, cw_step, jw,pol_columns,cat_col,Rx.copy())
      df_new = backtesting(Dx,Dc,pids,id_col,recs_df)
      mod_score_step = np.mean(df_new['MRR_4_score'])
      del df_new

      loss_step.append(1-mod_score_step) #1-score to mimic log loss where 0 means perfect performing model
      
  for i in range(0,len(jw)):
      jw_step = list(jw)
      jw_step[i] += step

      recs_df = recommender(6, cw, jw_step,pol_columns,cat_col,Rx.copy())
      df_new = backtesting(Dx,Dc,pids,id_col,recs_df)
      mod_score_step = np.mean(df_new['MRR_4_score'])
      del df_new
      
      
      loss_step.append(1-mod_score_step)
      

  return np.array(loss_step)
'''returns the change in loss accross all weightings'''

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''update for weighting with gradient descent method using the change in "loss" from the gradient function'''
def gradient_descent( cw, jw,pol_columns_r,cat_col_r,Rx,Dx,Dc,id_col,step):
    recs_df = recommender(6, cw, jw,pol_columns_r,cat_col_r,Rx.copy())
    pids = recs_df[id_col].drop_duplicates() #pids that have recommendations, training pids

    df_new = backtesting(Dx,Dc,pids,id_col,recs_df) #backtesting on the current fold

    mod_score = np.mean(df_new['MRR_4_score'])
    del df_new
    
    loss_step = gradient(cw,jw,step,Rx, pol_columns_r, cat_col_r, id_col,Dx,Dc,pids) #loss for each weighting
    d_loss = loss_step - np.full_like(loss_step, 1- mod_score)#differnce in loss

    cw_next = cw - d_loss[0:len(cw)]
    jw_next = jw - d_loss[len(cw):]
    
    return cw_next, jw_next, np.linalg.norm(d_loss)
    '''return the updated weigtings and the loss difference'''
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''runs k folds of training sets (some sample of backdated data) through recommender, and backtests each'''
def k_folds_optimization(df_x,k,Dx,Dc,recs,step,purpose,fold_size = 10000,fold_sample_size = 1000):


    scores = [] #list of scores for each fold
    cw_weightings = [] #list of cosine weightings used for each fold
    jw_weigtings = [] #list of jaccard weightings used for each fold

    #training data (subset of backdated data)
    pol_columns_r,cat_col_r, Rx,id_col = prep(df_x.copy().drop(columns = ['businessentityname', 'trn_grp_desc', 'prod_house_desc',
           'prod_tbl_grp_desc', 'psm_opt_desc',  'sky_pol_type_desc', 'product_desc', 'prod_tbl_cd', 'pa_sts_desc','business_only_indicator', 
           'death_benefit_indicator','pps_full_flag'])) 
    #pol_columns_r,cat_col_r, Rx,id_col = prep(df_x.copy())
    #columns dropped because of previous model's target leakage, but model performs better without these columns...
    #Maybe investigate effect of keeping/dropping certain columns

    Rx = Rx.drop(columns = ['pol_count','no unique products','set of policies']).set_index(id_col)
    
    for i in range(k):
        e = 0.1 #error value for gradient descent
        t = 70 #to avoid infinite loop

        Rx_s = Rx.copy().sample(fold_sample_size) #the fold of data used for optimization

        cw = (1)*np.array([1] *(Rx_s.shape[1] -cat_col_r)) 
        jw = (1)*np.array([1] *(cat_col_r)) #include policy info

        #--------------------optimizations of weightings------------------
        #process of finding optimal weightings on fold sample

        while e > 0.065 and t>0: #as e tend to zero the change in the "loss" value tends to zero
            cw,jw,e = gradient_descent(cw,jw,pol_columns_r,cat_col_r,Rx_s,Dx,Dc,id_col,step) 
            print(e, t, i)
            t = t-1 

        #------------------------------------------------------------------------
        #found optimal weigthings applied to the full fold for final fold scoring
        Rx_f = recommender(recs, cw, jw,pol_columns_r,cat_col_r,Rx.copy().sample(fold_size) )
        pids = Rx_f[id_col].drop_duplicates() #pids that have recommendations, training pids

        df_new = backtesting(Dx,Dc,pids,id_col,Rx_f.copy())

        scores.append(np.mean(df_new['MRR_4_score']))
        cw_weightings.append(cw)
        jw_weigtings.append(jw)

    if purpose == 'step size evaluation':
        return scores
    else:
        return df_new,cw_weightings,jw_weigtings,scores

'''returns the latest fold's dataframe with recs, the optimal weightings for all folds and the ave MRR_4 model score for all folds'''
