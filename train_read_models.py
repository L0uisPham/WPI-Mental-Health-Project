""" Functions for iterating and saving models """

from run_models import run_unstratified_BC, run_stratified_BC
from analysis import analyze_stratified_results, analyze_unstratified_results
from utils import *
from filtering import return_datasets
from run_settings import RUN_CONFIGS
from models import MODELS


def train_all_stratified_models(models=['RFC'],
        study_dir_name=None,
        version='',
        run_config='runs_post',
        top_features=None):
    #Train these models
    strats = ["Gender", "Race / Ethnicity", "GR"]

    for model_name, model_info in MODELS.items():
        if model_name not in models: continue
        print(model_name)
        clf = model_info['clf']
        hp = model_info['hyperparameters']
        
        for i,strat in enumerate(strats):
            strat_name = {'Gender':'gender','Race / Ethnicity':'race','GR':'gr'}[strat]

            fname = f'{version}{model_name}_{strat_name}'
            results = run_stratified_BC(return_datasets(post_only=True),
                RUN_CONFIGS[run_config],
                clf,hp,
                strat=strat,
                top_features=top_features[strat] if top_features else None) 
            save_results(results, name=fname, study=study_dir_name, run_type='stratified')


def read_all_stratified_models(models=['RFC'],
        study_dir_name=None,
        version='',
    ):
    #Train these models
    strats = ["Gender", "Race / Ethnicity", "GR"]
    
    
    for model_name, model_info in MODELS.items():
        if model_name not in models: continue
        dfs = []
        clf = model_info['clf']
        hp = model_info['hyperparameters']
        stratified_fc = {}
        
        for i,strat in enumerate(strats):
            strat_name = {'Gender':'gender','Race / Ethnicity':'race','GR':'gr'}[strat]
            fname = f'{version}{model_name}_{strat_name}'
            results = load_results(name=fname,study=study_dir_name,run_type='stratified')
            dfs.append(stratified_results_to_dataframe(results))
        
        total_df = pd.concat(dfs)

        save_df(total_df, name=f'{version}{model_name}_all_results',dir_name=study_dir_name)
    return total_df


def read_all_unstratified_models(models=['RFC'],
        study_dir_name=None,
        version='',
    ):
    #Train these models
    strats = ["Gender", "Race / Ethnicity", "GR"]
    
    
    for model_name, model_info in MODELS.items():
        if model_name not in models: continue
        clf = model_info['clf']
        hp = model_info['hyperparameters']
        stratified_fc = {}
        
        fname = f'{version}{model_name}'
        results = load_results(name=fname,study=study_dir_name,run_type='unstratified')
        total_df =  unstratified_results_to_dataframe(results)

        save_df(total_df, name=f'{version}{model_name}_all_results',dir_name=study_dir_name)
    return total_df



def get_fc_stratified_models(models=['RFC'],
        study_dir_name=None,
        version='',
    ):
    ''' Get feature importance for stratified models '''
    strats = ["Gender", "Race / Ethnicity", "GR"]
    
    
    for model_name, model_info in MODELS.items():
        if model_name not in models: continue
        dfs = []
        clf = model_info['clf']
        hp = model_info['hyperparameters']
        stratified_fc = {}
        
        for i,strat in enumerate(strats):
            print(strat)
            strat_name = {'Gender':'gender','Race / Ethnicity':'race','GR':'gr'}[strat]
            fname = f'{version}{model_name}_{strat_name}'
            results = load_results(name=fname,study=study_dir_name,run_type='stratified')
            if model_name == 'RFC':
                stratified_fc[strat] = analyze_stratified_results(results,study_dir_name=study_dir_name)
                print(stratified_fc[strat])
            else:
                assert 'Error: Please use RFC for FC analysis'
    return stratified_fc


def get_fc_unstratified_models(models=['RFC'],
        study_dir_name=None,
        version='',
    ):
    ''' Get feature importance for unstratified models '''
    strats = ["Gender", "Race / Ethnicity", "GR"]
    
    for model_name, model_info in MODELS.items():
        if model_name not in models: continue
        print(f'Training model type: {model_name}' )
        print(f'---------------------------------')
        
        clf = model_info['clf']
        hp = model_info['hyperparameters']
        df = return_datasets(post_only=True)
        
        fname = f'{version}{model_name}'
        results = load_results(name=fname,study=study_dir_name,run_type='unstratified')
        if model_name == 'RFC':
            fc = analyze_unstratified_results(results,study_dir_name=study_dir_name)
            # print(stratified_fc[strat])
        else:
            assert 'Error: Please use RFC for FC ananlysis'
    return fc


#changed from runs_post to runs-all since runs-post does not exist in the run-configs dictionary
def train_all_unstratified_models(models=['RFC'],study_dir_name=None,version='',run_config='runs_all'):
    ''' Train Unstratified models '''
    print(f'================================')
    print(f'UNSTRATIFIED TRAINING' )
    print(f'================================' )
    for model_name, model_info in MODELS.items():
        if model_name not in models: continue
        print(f'Training model type: {model_name}' )
        print(f'---------------------------------')
        
        clf = model_info['clf']
        hp = model_info['hyperparameters']
        df = return_datasets(post_only=True)
        run_configs=RUN_CONFIGS[run_config]
        handle_gr='G/R'
        fname = f'{version}{model_name}'
        results = run_unstratified_BC(df,run_configs,clf,hp)
        print('Saving results ...')
        save_results(results, name=fname,study=study_dir_name,run_type='unstratified')

