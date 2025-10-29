import pandas as pd

def data_preparation(data,idColumnName,itemColumnName):
    #make id column the first column
    idCol = data.pop(idColumnName)
    data.insert(0,idColumnName,idCol)
     # Count the number of items per user
    data['item_count'] = 1
    temporaryData = data.groupby(idColumnName)['item_count'].sum().reset_index()
    # Merge item count back into the original DataFrame
    data = data.drop(columns={'item_count'})
    data = pd.merge(data, temporaryData, 
                         on=idColumnName, 
                         how='left')
    # Create a set of unique items per user
    temporaryData = pd.DataFrame(data.groupby(idColumnName)[itemColumnName].apply(set))
    temporaryData = temporaryData.rename(columns={itemColumnName: 'set of items'})  #column containing a set of all item types client has

    data = pd.merge(data, temporaryData,on = idColumnName, how = 'left')

    # Count unique item types per user
    data['no. unique items'] = data['set of items'].apply(lambda x: len(x) if isinstance(x, set) else 0)

    # Process numerical features
    numericalData = data.iloc[:,1:].select_dtypes(include=['float64','int64']).copy() #creating numerical features data
    numericalData.insert(0,idColumnName,data[idColumnName])
    numericalData.iloc[:,1:] = numericalData.iloc[:,1:].fillna(0.0)

    def custom_agg(series): #if the numerical value is the same accross all rows per client don't take sum, take value (eg, how many items), otherwise take sum (eg. premium for policy)
        return series.iloc[0] if series.nunique() == 1 else series.sum()
    # Group by 'id' and apply the aggregation
    numericalData = numericalData.groupby(idColumnName).agg({col: custom_agg for col in numericalData.columns[1:]}).reset_index()

    #creating categorical feature data
    categoricalData = data.iloc[:,1:-2].select_dtypes(include=['object','bool','category']).copy()
    categoricalData.insert(0,idColumnName,data[idColumnName])
    categoricalData = pd.get_dummies(categoricalData.iloc[:,1:],columns = categoricalData.iloc[:,1:].columns, dtype=int,prefix_sep="_:")
    categoricalData.insert(0,idColumnName,data[idColumnName])
    categoricalData = categoricalData.groupby(idColumnName).sum()
    categoricalData = categoricalData[categoricalData.columns].map(lambda x: 1 if x > 0 else 0)

    data = data[[idColumnName, 'set of items']]
    # Convert set columns to sorted tuples so they're hashable and comparable
    data['set of items'] = data['set of items'].apply(
    lambda x: tuple(sorted(x)) if isinstance(x, set) else tuple())
    data = data.drop_duplicates()
    data['set of items'] = data['set of items'].apply(set)

    #merging categorical and numerical aggregated data
    aggregatedData = pd.merge(categoricalData, numericalData,on = idColumnName, how = 'inner') #aggregated dataframe
    aggregatedData = pd.merge(aggregatedData, data,on = idColumnName, how = 'inner')
    nCategoricalColumns = len(categoricalData.columns) #how many categorical columns there are

    itemColumns = [col for col in aggregatedData.columns if col.startswith(itemColumnName)] #list of all item columns

    return itemColumns,nCategoricalColumns, aggregatedData
'''returns item column's names, number of categorical columns, the aggregated dataframe and the id column name'''
