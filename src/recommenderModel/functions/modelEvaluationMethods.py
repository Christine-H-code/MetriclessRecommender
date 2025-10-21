import numpy as np
import pandas as pd

'''Function to calculate the mean reciprical rank score based on the recommendations and the new product taken out.'''

def MRR3(dataFrame,itemColumn): #scoring based on where new product sits in regards to recommendations, mean reciprical rank_3 and 4
 
    conditions = [dataFrame.apply(lambda row: len({row['no.1 most recommended item']}.intersection(row[itemColumn])) ,axis=1) == 1,
                  dataFrame.apply(lambda row: len({row['no.2 most recommended item']}.intersection(row[itemColumn])) ,axis=1) == 1, 
                  dataFrame.apply(lambda row: len({row['no.3 most recommended item']}.intersection(row[itemColumn])) ,axis=1) ==1]
    scores = [1, 0.66, 0.33]

    dataFrame['MRR_3_score'] = np.select(conditions, scores, default=0)

    

    return dataFrame

'''df is the dataframe given with MRR scores appended'''

def MRR4(dataFrame,itemColumn):

    dataFrame['MRR_4_score'] = [
    1 if f_rec in s else 0.75 if s_rec in s else 0.5 if t_rec in s else 0.25 if fr_rec in s else 0
    for f_rec,s_rec,t_rec,fr_rec,s in zip(dataFrame['no.1 most recommended item'],dataFrame['no.2 most recommended item'],dataFrame['no.3 most recommended item'],dataFrame['no.4 most recommended item'],dataFrame[itemColumn])]

    return dataFrame