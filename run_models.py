''' Run the models '''


import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer

# Supress Warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
pd.options.mode.chained_assignment = None 

from models import MODELS
from run_settings import RUN_CONFIGS
from filtering import return_datasets
import datetime

def run_unstratified_BC(df,run_configs,clf,hp,
        handle_gr='G/R', #choose from ['G/R', 'G', 'R', 'GR']
    ):
    all_results = {}
    for run_name, (features,target) in run_configs.items():
        print(f'- Running: {run_name}')
        #handle gr
        if handle_gr == 'G/R':
            gr = ['Gender', 'Race / Ethnicity']
        elif handle_gr == 'G':
            gr = ['Gender']
        elif handle_gr == 'R':
            gr = ['Race / Ethnicity']
        elif handle_gr == 'GR':
            gr = ['GR']
        df_sub = df.copy()
        for strat in gr:
            df_sub[strat] = df_sub[strat].apply(lambda x: {
                'Race / Ethnicity': lambda x: {'A':0,'W':1}[x],
                'Gender': lambda x: {'M':0,'F':1}[x],
                'GR': lambda x: x
            }[strat](x))
        X = df_sub[features + gr].to_numpy()
        y = df_sub[target].to_numpy()
        results, results_avg, best_params, models, outer_data = run_binary_classifier(X, y, clf, hyperparameters=hp)
        #TODO: Save more than just results here
        #put results into dict
        all_results[run_name] = {
            'name': run_name, 
            'model_name': clf.__name__,
            'gr_handled_as': handle_gr,
            'date_ran':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results':results,
            'results_avg':results_avg,
            'best_params':best_params,
            'models':models,
            'outer_data':outer_data,
            'features':features,
            'target':target,
            'X': X,
            'y': y,
        }
    return all_results



def run_stratified_BC(df, 
        run_configs, 
        clf, 
        hyperparams, 
        strat='Gender',
        top_features=None,):
    """
    df: dataframe to use
    run_configs: takes form of "{run name: (features, target)}
    clf: classifier to use (use class name, not instance)
    hyperparams: dict of hyperparams
    strat: Which to stratify by
    
    Returns a dictionary of lists of run_results (one per strat)
    """
    
    stratified_results = {}

    df['GR'] = df['Gender'] + df['Race / Ethnicity']

    for params_name, (features, target) in run_configs.items():
        stratified_results[params_name] = []
        # Loop over each run
        print(f'Running {params_name}')
        for strat_iter in set(df[strat]):
            
            #Problem: multiple strats are not saved
            try:
                print(top_features[params_name][strat])
            except:
                assert 'Error: Incompatible parameter set with top features'
            
            #stratification
            non_strat = {'Gender':'Race / Ethnicity', 'Race / Ethnicity': 'Gender', 'GR': 'GR'}[strat]
            
            df_sub = df[df[strat] == strat_iter].copy()

            if 'Race / Ethnicity' in features:
                features.remove('Race / Ethnicity')
            if 'Gender' in features:
                features.remove('Gender')

            df_sub = df_sub[features + [target, non_strat]]
            df_sub[non_strat] = df_sub[non_strat].apply(lambda x: {
                'Race / Ethnicity': lambda x: {'A':0,'W':1}[x],
                'Gender': lambda x: {'M':0,'F':1}[x],
                'GR': lambda x: x
            }[non_strat](x))


            if top_features != None:
                feat_set = [x[0] for x in top_features[params_name][strat_iter]['bio'] + top_features[params_name][strat_iter]['mh']]
                features = sorted([f for f in df_sub.columns if f in feat_set])

            df_sub = df_sub[features + [non_strat, target]]

            if 'GR' in df_sub.columns: 
                df_sub = df_sub.drop(columns=['GR'])            
                X = df_sub[features].to_numpy()
                y = df_sub[target].to_numpy()
            else:
                X = df_sub[features + [non_strat]].to_numpy()
                y = df_sub[target].to_numpy()
            
            #run the BC
            try:
                results, results_avg, best_params, models, outer_data = run_binary_classifier(X, y, clf, hyperparameters=hyperparams)
                broken = False
            except:
                broken = True

            run_results = {
                'name': params_name, 
                'model_used': clf.__name__,
                'strat': strat,
                'strat_iter': strat_iter,
                'date_ran': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'broken': broken,
                'uses_top_features': top_features != None,
                'target': target if not broken else None,
                'features': features if not broken else None,
                'results': results if not broken else None,
                'results_avg': results_avg if not broken else None,
                'best_params': best_params if not broken else None,
                'models': models if not broken else None,
                'outer_data': outer_data if not broken else None,
                'X': X if not broken else None,
                'y': y if not broken else None,
            }
            stratified_results[params_name].append(run_results)

    return stratified_results

def tune_hyperparameters(clf, X_train, y_train, hyperparameters,verbose=0):
    ''' Tune HP helper '''

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    grid_search = GridSearchCV(clf, hyperparameters, cv=cv, verbose=0, n_jobs=3)
    print(f'result: {grid_search.fit(X_train, y_train)}')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def run_binary_classifier(X, y, clf, params=None, hyperparameters=None,suppress_print=False,verbose=0):
    ''' Run BC with stratified 5-fold cross validation'''
    results = {
        'train_acc': [],
        'train_F1': [],
        'test_acc': [],
        'test_F1': [],
        'val_acc': [],
        'val_F1': [],
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    split_indices = sss.split(X, y)
    models = []
    outer_data = []
    for i, (train_index, test_index) in enumerate(split_indices):
        if not suppress_print: print(f"\nSplit {i+1} ...")
        
        if params is None:
            if not suppress_print: print('Tuning hyperparameters ...')
            best_params = tune_hyperparameters(clf(), X[train_index], y[train_index], hyperparameters,verbose=verbose)
        else:
            best_params = params
        inner_data = {'X_val_train': [], 'y_val_train': [], 'X_val': [], 'y_val': [], 'X_train': [], 'y_train': [], 'X_test': [], 'y_test': []}
        inner_data['X_test'].append(X[test_index])
        inner_data['y_test'].append(y[test_index])
        inner_data['X_train'].append(X[train_index])
        inner_data['y_train'].append(y[train_index])
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        for j, (val_train_index, val_index) in enumerate(cv.split(X[train_index], y[train_index])):
            X_val_train, X_val = X[val_train_index], X[val_index]
            y_val_train, y_val = y[val_train_index], y[val_index]

            inner_data['X_val_train'].append(X_val_train)
            inner_data['X_val'].append(X_val)
            inner_data['y_val_train'].append(y_val_train)
            inner_data['y_val'].append(y_val)

            model = clf(**best_params)
            model.fit(X_val_train, y_val_train)
            # Train accuracy
            y_pred_train = model.predict(X_val_train)
            train_acc = accuracy_score(y_val_train, y_pred_train)
            train_F1 = f1_score(y_val_train, y_pred_train, average='macro')
            # Validation accuracy
            y_pred_val = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_F1 = f1_score(y_val, y_pred_val, average='macro')
            results['train_acc'].append(train_acc)
            results['train_F1'].append(train_F1)
            results['val_acc'].append(val_acc)
            results['val_F1'].append(val_F1)

        model = clf(**best_params)
        model.fit(X[train_index], y[train_index])

        # Test accuracy
        y_pred_test = model.predict(X[test_index])
        test_acc = accuracy_score(y[test_index], y_pred_test)
        test_F1 = f1_score(y[test_index], y_pred_test, average='macro')

        results['test_acc'].append(test_acc)
        results['test_F1'].append(test_F1)

        models.append(model)
        outer_data.append(inner_data)

    
    val_acc = np.mean(results['val_acc'])
    val_F1 = np.mean(results['val_F1'])
    train_acc = np.mean(results['train_acc'])
    train_F1 = np.mean(results['train_F1'])
    test_acc = np.mean(results['test_acc'])
    test_F1 = np.mean(results['test_F1'])

    results_avg = {
        'val_acc': val_acc,
        'val_F1': val_F1,
        'train_acc': train_acc,
        'train_F1': train_F1,
        'test_acc': test_acc,
        'test_F1': test_F1,
    }

    return results, results_avg, best_params, models, outer_data