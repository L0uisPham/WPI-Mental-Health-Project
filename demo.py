#!/usr/bin/env python
# coding: utf-8

# # WHS Analysis Notebook
# 
# Helpful files
# - *Constants.py*: Store constants
# - *Filtering.py*: Filter the data for use
# - *utils.py*: Helper functions for saving things
# 
# ML Files
# - *Analysis.py*: Get feature contributions
# - *Train_read_models.py*: Functions to train stratified or unstratified
# - *run_models.py*: Functions to specifically run the models
# - *run_settings.py*: Features to use for models
# - *models.py*: Specific models to use with hp configs

# ## 1 Statistical Analysis

# ### 1.1 Summary Statistics

# In[1]:


from constants import *
import scipy.stats as stats
from filtering import return_datasets
import numpy as np
import pandas as pd

df = return_datasets(post_only=True)


gad_cols = ['GAD 7 Risk'] + GAD7_Q_NAMES
phq_cols = ['PHQ 9 Risk'] + PHQ9_Q_NAMES
regular_cols = [x for x in CORE_FEATURES if x not in ['Gender','Race / Ethnicity']] + ['GPA Weighted','GPA Unweighted']

#Place mean and std of each col in a table based on stratification of gender
def make_table(df, cols):
    df = df.copy()
    df['GR'] = df['Gender'] + df['Race / Ethnicity']
    normal_df = df.copy()[sorted(set(cols + ['Race / Ethnicity','Gender','GR']))]

    # Loop strats
    dfs = []  
    strats = ['Race / Ethnicity','Gender','GR']
    for s in strats:
        # Calculate mean and std for each group and column
        grouped_df = normal_df.drop(columns=[s_ for s_ in strats if s_ in normal_df.columns and s_!=s]).groupby(s).agg([np.mean, stats.sem])
        formatted_df = grouped_df.apply(lambda x: pd.Series([f"{x[col][0]:.2f} ± {x[col][1]:.2f}" for col in cols], index=cols), axis=1)

        #adjust index for counts
        new_index = [f'{x} ({normal_df[s].value_counts().loc[x]})' for x in list(formatted_df.index)]
        formatted_df.index = new_index

        dfs.append(formatted_df) #append
    
    dfs = pd.concat(dfs) #compile
    return dfs


#TODO: Export tables
gad_table = make_table(df,gad_cols)
phq_table = make_table(df,phq_cols)
reg_table = make_table(df,regular_cols)

from utils import *

dir_name='April_18'
# save_df(gad_table, 'gad_table',dir_name=dir_name)
# save_df(phq_table, 'phq_table',dir_name=dir_name)
# save_df(reg_table, 'reg_table',dir_name=dir_name)
print(reg_table)
print(phq_table)
print(gad_table)
# gad_table.to_csv(f'{}{dir_name}{}')


# ### 1.2 ANOVA

# In[2]:


import pingouin as pg

run_type='Outcome'

def run_ANOVA(run_type):
    ''' Run ANOVA for Outcome, GAD7, or PHQ9 '''
    df = return_datasets(post_only=True)
    q_names = {
        'Outcome': ['PHQ 9 Risk', 'GAD 7 Risk', 'Endorse Q9'],
        'GAD7':GAD7_Q_NAMES + ['GAD 7 Risk'],
        'PHQ9':PHQ9_Q_NAMES + ['PHQ 9 Risk'],
    }[run_type]

    p_unc_list = []
    for i,q in enumerate(q_names):
        model1 = pg.anova(dv=q, between=['Gender','Race / Ethnicity'], data=df, detailed=True)
        print(model1)    
        col= model1['p-unc']
        col = col.iloc[0:3]
        col.name = q

        p_unc_list.append(col)

    df_tabbed = pd.concat(p_unc_list, axis=1)
    df_tabbed = df_tabbed.rename(index={0:'Gender (ME)', 1:'Race (ME)', 2:'Race/Gender (IE)'})
    def format_pvalue(p_value):
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = ""
        return f"{p_value:.2e} {significance}"

    df_tabbed = df_tabbed.applymap(format_pvalue)
    print(df_tabbed)

run_ANOVA('Outcome')
# run_ANOVA('GAD7')
# run_ANOVA('PHQ9')


# ## 2 Train / Read models

# In[ ]:


from run_models import *
from analysis import *
from utils import *
from train_read_models import *

dir_name='April_18'

#train models
train_all_unstratified_models(
    models=['RFC'],
    study_dir_name=dir_name,
    version='summary',
)


#------------------------------------------
# Need to fix some bugs here
#------------------------------------------

#extract results
total_df = read_all_unstratified_models(
    models=['RFC'],
    study_dir_name=dir_name,
    version='question',
)


# ## 3 Feature Contribution Analysis

# In[ ]:


from constants import EDU_FEATURES, PHQ9_Q_NAMES, GAD7_Q_NAMES

#sort and get rank
def parse_fc(fc,top_n=3,container=None):
    #convert to rank list
    if container is not None:
        fc = [x for x in fc if x[0] in container]
    fc = sorted(fc, key=lambda x: x[1],reverse=True)
    fc = [(x[0],i) for i,x in enumerate(fc)]
    
    return fc[0:top_n] #get top features

#sort and get rank
def parse_fc_2(fc, top_n=2, container=None):
    #convert to rank list
    if container is not None:
        fc = [x for x in fc if x[0] in container]
    
    fc = sorted(fc, key=lambda x: x[1],reverse=True)
    #score features
    fc = [(x[0],top_n-i) for i,x in enumerate(fc) if i < top_n]
    return fc[0:top_n] #get top features

def flatten(nl):
    return [item for sublist in nl for item in sublist]

def parse_title(t,s=None,top_n=2):
    target = t.split(':')[0]
    if s: 
        strat = {'M':'Male',
                'F':'Female',
                'A':'Asian',
                'W':'White',
                'MW':'White Male',
                'MA':'Asian Male',
                'FA': 'Asian Female',
                'FW': 'White Female'}[s]
        return f'Average Feature contributions in predicting {target} scores \n  in Student {strat} Population'
    else:
        return f'Average Feature contributions in predicting {target} scores \n  in Total Student Population' 

top_n_mh = 2
top_n_bio = 2
top_features = {}
import tqdm

mh_table = []
bio_table = []


def parse_fc_3(fc, container):
    #flatten list fc
    fc = flatten(fc)
    return [x for x in fc if x[0] in container]

#summary level
def run_summary_level_unstratified():
    for key_param, val_param in tqdm.tqdm(us_fc_summary.items()): #strat
        if 'PHQ' in key_param:
            mh_features = ['GAD 7 Risk Binary', 'Endorse Q9 Risk'] + EDU_FEATURES
        elif 'GAD' in key_param:
            mh_features = ['PHQ 9 Risk Binary', 'Endorse Q9 Risk'] + EDU_FEATURES
        elif 'Endorse Q9' in key_param:
            mh_features = ['GAD 7 Risk Binary', 'PHQ 9 Risk Binary'] + EDU_FEATURES
        title = parse_title(f'{key_param}', top_n="Top 2 Mental Health")
        
        #TODO: line does nothing
        top_fc_bio = parse_fc_3(val_param,container=mh_features)
        
        def avg_features(feature_name,features_list):
            print([x[1] for x in features_list if x[0] == feature_name])
            return sum([x[1] for x in features_list if x[0] == feature_name])/5
        
        avgs_fc_bio = [(f,avg_features(f,top_fc_bio)) for f in mh_features]
        avgs_fc_bio = sorted(avgs_fc_bio, key=lambda x: x[1],reverse=True)

        plt.figure(figsize=(8,5))
        sns.barplot(y=[x[0] for x in avgs_fc_bio], x=[x[1] for x in avgs_fc_bio], palette='viridis',orient='h')
        # plt.title(title,weight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

GAD_NAME = {
    'GAD7-1': 'Anxious',
    'GAD7-2': 'Worrying',
    'GAD7-3': 'Worrying',
    'GAD7-4': 'Stress',
    'GAD7-5': 'Restless',
    'GAD7-6': 'Irritable',
    'GAD7-7': 'Fear'
}

PHQ_NAME = {
    'PHQ9-1': 'Anhedonia',
    'PHQ9-2': 'Hopeless',
    'PHQ9-3': 'Sleep',
    'PHQ9-4': 'Fatigue',
    'PHQ9-5': 'Appetite',
    'PHQ9-6': 'Failure',
    'PHQ9-7': 'Concentration',
    'PHQ9-8': 'Slow speaking / Restless',
    'PHQ9-9': 'Suicidal'

}

#question level
def run_question_level_unstratified():
    for key_param, val_param in tqdm.tqdm(us_fc_question.items()): #strat
        print(key_param)
        if 'PHQ' in key_param or 'GAD' in key_param:
            mh_features = GAD7_Q_NAMES + PHQ9_Q_NAMES
        elif 'Endorse Q9' in key_param:
            mh_features = GAD7_Q_NAMES + [q for q in PHQ9_Q_NAMES if '-9' not in q]
        title = parse_title(f'{key_param}', top_n="Top 2 Mental Health")
        top_fc_bio = parse_fc_3(val_param,container=mh_features)
        
        def avg_features(feature_name,features_list):
            return sum([x[1] for x in features_list if x[0] == feature_name])/5
            
        avgs_fc_biio = [(f,avg_features(f,top_fc_bio)) for f in mh_features]
        avgs_fc_bio = sorted(avgs_fc_bio, key=lambda x: x[1],reverse=True)
        
        fig,ax = plt.subplots(figsize=(8,5))

        def get_q_label(q):
            l = GAD_NAME[q] if 'GAD' in q else PHQ_NAME[q]
            return l

        color_mapping = {
            **{f'PHQ9-{x}': '#f25c54' for x in range(1,10)},
            **{f'GAD7-{x}': '#43aa8b' for x in range(1,8)}
        }
        mh_pallette = sns.color_palette(color_mapping.values())

        sns.barplot(y=[x[0] for x in avgs_fc_bio], x=[x[1] for x in avgs_fc_bio],palette= color_mapping ,orient='h', ax=ax)
        new_labels = [get_q_label(q.get_text()) for q in ax.get_yticklabels()]
        
        for i,l in enumerate(new_labels):
            x_min, x_max = plt.xlim()
            annot_buffer =  0.01 * (x_max - x_min)
            ax.annotate(l, xy=(annot_buffer+avgs_fc_bio[i][1],i),va='center',fontsize=12)
        
        #modify spacing
        x_min, x_max = plt.xlim()
        x_buffer = 0.15 * (x_max - x_min)
        plt.xlim(x_min, x_max + x_buffer)

        y_min, y_max = plt.ylim()
        y_buffer = 0.015 * (y_max - y_min)
        plt.ylim(y_min-y_buffer, y_max + y_buffer)
        # plt.title(title,weight='bold',fontsize=20)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

run_question_level_unstratified()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




