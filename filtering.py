'''Process the data '''

import pandas as pd
import numpy as np

from helpers import (
    phq9_map,
    gad7_map,
    endorseQ9_map,
    sped_map,
    ell_map,
    planstatus_map,
)

PATH_TO_PRE_DATA = 'data/WPI_MH_Data_Pre.csv'
PATH_TO_POST_DATA = 'data/WPI_MH_Data_Post.csv'

def filter_datasets(dfs):

    for df in dfs:
        df.dropna(axis=0, inplace=True, how='any')
        df.reset_index(drop=True, inplace=True)
        ''' Make sure to run analysis with only using these labels on targets (or do separate analyses'''
        df['GAD 7 Risk Binary'] = df['GAD 7 Risk'].apply(lambda x: gad7_map(x))
        df['PHQ 9 Risk Binary'] = df['PHQ 9 Risk'].apply(lambda x: phq9_map(x))
        df['Endorse Q9 Binary'] = df['Endorse Q9'].apply(lambda x: endorseQ9_map(x))

        df['GR'] = df['Gender'] + df['Race / Ethnicity']
        
        df['504 Plan Status'] = df['504 Plan Status'].apply(lambda x: planstatus_map(x))
        df['ELL status'] = df['ELL status'].apply(lambda x: ell_map(x))
        df['SPED status'] = df['SPED status'].apply(lambda x: sped_map(x))
        
        df['GAD 7 Risk Multinary'] = df['GAD 7 Risk'].apply(lambda x: gad7_map(x,'default'))
        df['PHQ 9 Risk Multinary'] = df['PHQ 9 Risk'].apply(lambda x: phq9_map(x,'default'))
    
    return dfs

def get_pre_post_data(dfs):
    df_pre, df_post = dfs

    #TODO: Abstract
    pre_col_mapper = {
        'GPA Weighted':'GPA Weighted S1',
        'GPA Unweighted':'GPA Unweighted S1',
    }
    post_col_mapper = {
        'GPA Weighted':'GPA Weighted YE',
        'GPA Unweighted':'GPA Unweighted YE',
    }

    df_pre.rename(pre_col_mapper, axis='columns', inplace=True)
    df_post.rename(post_col_mapper, axis='columns', inplace=True)
    total_df = df_post.merge(df_pre[['ID MAPPER', 'GPA Weighted S1', 'GPA Unweighted S1']], on=['ID MAPPER'])
    
    total_df['GPA Weighted Delta'] = total_df['GPA Weighted YE'] - total_df['GPA Weighted S1']
    total_df['GPA Unweighted Delta'] = total_df['GPA Unweighted YE'] - total_df['GPA Unweighted S1']


    total_df.dropna(axis=0, inplace=True, how='any')
    total_df.reset_index(drop=True, inplace=True)

    return total_df

def prune_df(df):
    df = df[df.groupby('Gender')['Gender'].transform('count')>10]
    df = df[df.groupby('Race / Ethnicity')['Race / Ethnicity'].transform('count')>10]
    return df

def return_datasets(pruned=True,post_only=False):
    if post_only: #return only gpa from YE
        df = pd.read_csv(PATH_TO_POST_DATA)
        df = prune_df(filter_datasets([df])[0])
    else:
        df = pd.read_csv(PATH_TO_PRE_DATA)
        df_pre = pd.read_csv(PATH_TO_POST_DATA)
        
        df, df_pre = filter_datasets([df, df_pre])
        df = get_pre_post_data([df, df_pre])
        
        if pruned: df = prune_df(df)

    return df

FEATURES_DF = return_datasets(post_only=True)

