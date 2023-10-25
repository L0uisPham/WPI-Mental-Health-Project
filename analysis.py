
from sklearn.metrics import classification_report


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from models import MODELS
from utils import parse_params_name_to_directory

def analyze_stratified_results(
        stratified_results,
        to_run='feature_contributions',
        study_dir_name=None,export=False):
    
    # Code to fit model and obtain feature importances
    stratified_fc = {}
    
    # Iterate through each set of parameters
    for params_name, params_results in stratified_results.items():
        stratified_fc[params_name] = {}
        
        # Iterate through each set of gender/race/both split results for the current set of parameters
        for strat_results in params_results:
            stratified_fc[params_name][strat_results['strat_iter']] = []
            
            if to_run == 'feature_contributions':
                #iter through each run
                for i in range(len(strat_results['outer_data'])):
                    
                    data = strat_results['outer_data'][i]
                    
                    X = strat_results['X']
                    y = strat_results['y']

                    strat_name = strat_results['strat']
                    results_avg = strat_results['results_avg']
                    results = strat_results['results']
                    group = {
                        'A':'Asian',
                        'W':'White',
                        'M':'Male',
                        'F':'Female',
                        'MW':'Male White',
                        'MA':'Male Asian',
                        'FW':'Female White',
                        'FA':'Female Asian',
                    }[strat_results['strat_iter']]

                    if strat_name != 'GR':
                        non_strat_name = {'Gender':'Race / Ethnicity', 'Race / Ethnicity': 'Gender'}[strat_name]
                        X_names = strat_results['features'] + [non_strat_name]
                    else:
                        X_names = strat_results['features']
                    targ_name = strat_results['target']
                    model = strat_results['models'][i]
                    
                    directory = parse_params_name_to_directory(params_name,study_dir_name=study_dir_name)
                    fc = feature_contributions(X_names, targ_name, model,
                        results_avg=results_avg,
                        results=results,
                        strat=strat_name, 
                        group=group,
                        directory=directory,
                        group_code=strat_results['strat_iter'], 
                        top_pct_trees=None, 
                        data=data,
                        run_i=i,
                        save_plots=True,
                        )
                    
                    stratified_fc[params_name][strat_results['strat_iter']].append(fc)

            # run_results = {
            #     'name': run_name, 
            #     'strat': strat,
            #     'strat_iter': strat_iter,
            #     'target': target,
            #     'features': features,
            #     'results': results,
            #     'results_avg': results_avg,
            #     'best_params': best_params,
            #     'models': models,
            #     'outer_data': outer_data
            # }
    return stratified_fc

def analyze_unstratified_results(
        results,
        to_run='feature_contributions',
        study_dir_name=None,export=False):#X, y, x_names, targ_name, model, strat=None, top_pct_trees=None, data=None, run_i=0):
    
    all_fc = {}
    # Iterate through each set of parameters
    for params_name, params_results in results.items():
        # Iterate through each set of gender/race/both split results for the current set of parameters
        all_fc[params_name] = []
        if to_run == 'feature_contributions':
            #iter through each run
            for i in range(len(params_results['outer_data'])):
                
                data = params_results['outer_data'][i]
                
                X = params_results['X']
                y = params_results['y']

                results_avg = params_results['results_avg']
                results = params_results['results']
                
                X_names = params_results['features']
                targ_name = params_results['target']
                model = params_results['models'][i]

                directory = parse_params_name_to_directory(params_name,study_dir_name=study_dir_name)
                fc = feature_contributions(X_names, targ_name, model,
                    results_avg=results_avg,
                    results=results,
                    directory=directory,
                    top_pct_trees=None, 
                    data=data,
                    run_i=i,
                    save_plots=True,
                    plot=False,
                )
                all_fc[params_name].append(fc)
    return all_fc


def describe_data(data, targ_name, run_i, results_avg=None, strat=None, group=None,results=None):
    """Generate description of data for a given dataset."""
    
    targ_map = {
        'GAD 7 Risk Binary': lambda x: {0: "low", 1: "high"}[x],
        'PHQ 9 Risk Binary': lambda x: {0: "low", 1: "high"}[x],
        'Endorse Q9 Binary': lambda x: {0: "low", 1: "high"}[x]
    }[targ_name]

    n_train = len(data['X_train'][0])
    n_test = len(data['X_test'][0])
    n_total = n_train + n_test
    y_test_labeled = list(map(targ_map, data['y_test'][0]))
    y_train_labeled = list(map(targ_map, data['y_train'][0]))
    
    desc = (
        f'Target: {targ_name}',
        f'Stratified by: {strat}',
        f'Group: {group}',
        f'N Total: {n_total}',
        f'N Train: {n_train}',
        f'F1 Train: {results["train_F1"][run_i]}',
        f'Acc Train: {results["train_acc"][run_i]}',
        f'F1 Test: {results["test_F1"][run_i]}',
        f'Acc Test: {results["test_acc"][run_i]}',
        f'Train Target Dist.:{sorted(dict(Counter(y_train_labeled)).items(), key=lambda x:x[0])}',
        f'N Test: {n_test}',
        f'Test Target Dist.:{sorted(dict(Counter(y_test_labeled)).items(), key=lambda x:x[0])}'
    )
    
    return desc

def plot_feature_contributions(x_names, feature_importances, std, ax_feat, top_n_trees=None):
    """Plot feature contributions of a random forest classifier."""
    
    title = 'Feature Contributions'
    
    xy= list(zip(*dict(zip(x_names, feature_importances)).items()))
    feat_x = list(xy[0])
    feat_y = list(xy[1])
    
    forest_importances = pd.Series(feature_importances, index=x_names)
    forest_importances.plot.barh(xerr=std, ax=ax_feat)
    
    ax_feat.set_title(title, fontstyle='italic',fontsize=15)


def plot_description(desc, ax_desc):

    title = 'Description'
    ax_desc.set_title(title, fontstyle='italic',fontsize=15)

    for j, stat in enumerate(desc):
        scale_factor = 1 #defines how low the desciption goes. (1 is default, higher numbers shrink)
        bottom_padding = (len(desc)+1) * scale_factor
        top_padding =  1- (j+1)*(1/bottom_padding)
        left_padding = 0.05
        ax_desc.text(left_padding, top_padding, desc[j], fontsize=15)


def get_feature_contributions(data, model, x_names, top_pct_trees=None):
    # Calculate the feature contributions

    X = data['X_test'][0]
    y = data['y_test'][0]
   
    #handle tree performances
    tree_acc = []
    for j, tree_ in enumerate(model.estimators_):
        y_pred = tree_.predict(X)
        acc = accuracy_score(y, y_pred)
        F1 = f1_score(y, y_pred, average='macro')
        tree_acc.append((j, acc, F1))
    
    #handle top tres
    if top_pct_trees:
        n_estimators = int(len(model.estimators_) * pct_estimators)
        sorted_trees_idx = sorted(tree_acc, key=lambda x: x[1] + x[2], reverse=True)
        std = np.std([model.estimators_[idx].feature_importances_ for i, (idx, _, _) in enumerate(sorted_trees_idx) if i < n_estimators], axis=0)
        mu = np.mean([tree.feature_importances_ for tree in model.estimators_[:n_estimators]], axis=0)
    else:
        n_estimators = len(model.estimators_)
        std = np.std([model.estimators_[i].feature_importances_ for i in range(len(model.estimators_))])
        mu = np.mean([model.estimators_[i].feature_importances_ for i in range(len(model.estimators_))])
    
    mean_accuracy = np.mean([x[1] for x in tree_acc])
    mean_F1 = np.mean([x[2] for x in tree_acc])

    print(f'Top {n_estimators} of {len(model.estimators_)} Trees Used')
    feature_importances = model.feature_importances_#bF.best_estimator_.feature_importances_
    named_feature_importances = list(zip(x_names, feature_importances))

    if top_pct_trees:
        top_n_feats = int(top_pct_trees * n_estimators)
        print(f"Using {top_n_trees} of {n_estimators} estimators to compute feature importances")
        feat_imp = np.zeros(len(x_names))
        for tree_idx in range(top_n_trees):
            feat_imp += model.estimators_[sorted_trees_idx[tree_idx][0]].feature_importances_
        named_feature_importances = sorted(zip(x_names, feat_imp), key=lambda x: x[1], reverse=True)[:top_n_feats]
    else:
        top_n_feats = n_estimators
        named_feature_importances = sorted(named_feature_importances, key=lambda x: x[1], reverse=True)[:top_n_feats]

    return std, mu, named_feature_importances, mean_accuracy, mean_F1


def feature_contributions(x_names, targ_name, model, 
        results_avg=None,results=None, strat=None, 
        group=None, 
        directory=None,
        group_code=None,
        top_pct_trees=None, data=None,
        run_i=0,
        save_plots=False,
        plot=True):
    
    ''' Function to get feature contributions with optional plotting '''

    if isinstance(model, RandomForestClassifier) or isinstance(model, DecisionTreeClassifier):
        feature_importances = model.feature_importances_
    elif isinstance(model, XGBClassifier):
        feature_importances = model.feature_importances_
    else:
        raise ValueError('Unsupported model type: {}'.format(type(model)))
    
    
    std, mu, named_feature_importances, mean_accuracy, mean_F1 = get_feature_contributions(data=data, model=model, x_names=x_names, top_pct_trees=top_pct_trees)
    
    if plot == False:
        return named_feature_importances
    
    # Get data description
    desc = describe_data(data, targ_name, run_i, strat=strat,group=group,results_avg=results_avg,results=results)

    # Add axes
    fig, (ax_feat, ax_desc) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.8, 1]})
    fig.suptitle(f'Feature Contributions for {targ_name} in {group} Population', fontsize=20,weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.9) # Adjust the top margin
    
    # Plot axes
    plot_feature_contributions(x_names, feature_importances, std, ax_feat)
    plot_description(desc, ax_desc=ax_desc)
    
    # Adjust layout and show plot
    if save_plots:
        plt.tight_layout()
        if strat:
            plt.savefig(f'{directory}/RFC_g{group_code}_r{run_i+1}.png')
        else:
            plt.savefig(f'{directory}/RFC_g{group_code}.png')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
    
    return named_feature_importances
