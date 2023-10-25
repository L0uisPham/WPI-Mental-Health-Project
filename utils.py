""" Helper functions for saving/loading/exporting """

import os
import pandas as pd
import numpy as np
import pickle

STRATIFIED_RESULTS_DIR = 'results/'

def print_stratified_results(stratified_results):
    for i, (params_name, param_resulst) in enumerate(stratified_results.items()):  
        print(f"Stratification {i+1}:")
        print(f"Run Name: {run_results['name']}")
        print("Results Average:")
        print(run_results['results_avg'])
        print("Best Parameters:")
        print(run_results['best_params'])

def unstratified_results_to_dataframe(results, feature_contributions=None):
    data = []
    for run_name, run_results in results.items():

        # if feature_contributions:
        # if run_results['broken']: continue

        data_cols = {
            'Parameters Used': run_results['name'],
            'Target': run_results['target'],
            'Train n': len(run_results['outer_data'][0]['X_train'][0]),
            'Test n': len(run_results['outer_data'][0]['X_test'][0]),
            'Test F1': run_results['results_avg']['test_F1'],
            'Test Acc': run_results['results_avg']['test_acc'],
            'Val F1': run_results['results_avg']['val_F1'],
            'Val Acc': run_results['results_avg']['val_acc'],
            'Train F1': run_results['results_avg']['train_F1'],
            'Train Acc': run_results['results_avg']['train_acc'],
        }

        other_cols = {}
        
        # 'Features Used': run_results['features']
        data.append(data_cols)

    return pd.DataFrame(data)

def stratified_results_to_dataframe(stratified_results, feature_contributions=None):
    data = []
    for run_name in stratified_results:

        for i,run_results in enumerate(stratified_results[run_name]):
            # if feature_contributions:
            
            if run_results['broken']: continue

            data_cols = {
                'Parameters Used': run_results['name'],
                'Strat': run_results['strat'],
                'Target': run_results['target'],
                'Group': run_results['strat_iter'],#{'A':'Asian','W':'White','M':'Male','F':'Female'}[run_results['strat_iter']],
                'Train n': len(run_results['outer_data'][i]['X_train'][0]),
                'Test n': len(run_results['outer_data'][i]['X_test'][0]),
                'Test F1': run_results['results_avg']['test_F1'],
                'Test Acc': run_results['results_avg']['test_acc'],
                'Val F1': run_results['results_avg']['val_F1'],
                'Val Acc': run_results['results_avg']['val_acc'],
                'Train F1': run_results['results_avg']['train_F1'],
                'Train Acc': run_results['results_avg']['train_acc'],
            }
            
            other_cols = {}

            # 'Features Used': run_results['features']
            data.append(data_cols)
    
    return pd.DataFrame(data)

def save_df(df,
    name,
    dir_name,
    path=STRATIFIED_RESULTS_DIR,
    version='',
    ):

    if not os.path.isdir(f'{path}{dir_name}/'): os.mkdir(f'{path}{dir_name}/')
    df.to_csv(f'{path}{dir_name}/{name}.csv')

def save_results(stratified_results,
        name, 
        study='RFC',
        path=STRATIFIED_RESULTS_DIR,
        run_type='stratified',
        version='',
        ):
    if not os.path.isdir(f'{path}{study}'): os.mkdir(f'{path}{study}')
    with open(f'{path}{study}/{name}_{run_type}.pkl', 'wb') as f:
        pickle.dump(stratified_results, f)

def load_results(
        name, 
        study='RFC',
        path=STRATIFIED_RESULTS_DIR,
        run_type='stratified',
        version='',
    ):
    try:
        if not os.path.isdir(f'{path}{study}'): os.mkdir(f'{path}{study}')
        print(f'Path: {path}{study}/{name}_{run_type}.pkl')
        if name:
            with open(f'{path}{study}/{name}_{run_type}.pkl', 'rb') as f:
                stratified_results = pickle.load(f)

        else:
            print('"name" not provided, using generic name.')
            with open(f'{path}{study}/{run_type}.pkl', 'rb') as f:
                stratified_results = pickle.load(f)
    except:
        stratified_results = None
    return stratified_results

def export_stratified_table(stratified_results):
    info_cols={
        'Study Name': study_name,
        'Target':targ, 
        'Stratified by':study.stratify_cols, 
        'GPA Group': gpa_mode,
        'Subpopulation 1': label[0],
        'Subpopulation 2': sub_2,
        'full_n': study_size,
        'test_n': y_test.shape[0],
    }
    
    feature_cols = feature_cols
    results_cols = results_cols


def parse_params_name_to_directory(name,
        study_dir_name=None,
        path=STRATIFIED_RESULTS_DIR
    ):
    parsed_name = name.replace(':','').replace(' ','_')
    if study_dir_name:
        if not os.path.isdir(path): os.mkdir(path)
        if not os.path.isdir(f'{path}{study_dir_name}/'): os.mkdir(f'{path}{study_dir_name}/')
        if not os.path.isdir(f'{path}{study_dir_name}/{parsed_name}/'): os.mkdir(f'{path}{study_dir_name}/{parsed_name}/')
        ret = f'{path}{study_dir_name}/{parsed_name}/'
    else:
        if not os.path.isdir(parsed_name): os.mkdir(parsed_name)
        ret = f'{parsed_name}/'

    return ret