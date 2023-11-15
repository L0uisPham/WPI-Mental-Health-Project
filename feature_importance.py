import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
grade = 'Grade 7' 
data = pd.read_csv(f'new_data/{grade}.csv')
moderate_threshold = 40
high_threshold = 70

phq_moderate = 10.0
phq_severe = 20.0

gad_moderate = 10.0
gad_severe = 15.0

mode = {
    0: 'Depression T Score',
    1: 'Anxiety T Score',
    2: 'PHQ-9 Total',
    3: 'GAD-7 Total',
    4: 'Total T Score'
}
target = mode[0]

score = mode[2] 


# Encode categorical variables 
le = LabelEncoder()

X = data[['Unweighted GPA', 'Endorse Q9', 'Absence', 'Tardy', 'Dismissal', 'Gender', 'Race / Ethnicity', 'ELL', 'SPED', '504']]
data['target_column'] = pd.cut(data[score], bins=[0, phq_moderate, phq_severe, float('inf')],
                               labels=['Low', 'Moderate', 'High'], right=False)

Y = data['target_column']

X = pd.get_dummies(X, columns=['Gender'], prefix=['Gender'])

X = pd.get_dummies(X, columns=['Race / Ethnicity'], prefix=['Race'])

X = pd.get_dummies(X, columns=['SPED'], prefix=['SPED'])

X = pd.get_dummies(X, columns=['ELL'], prefix=['ELL'])

X = pd.get_dummies(X, columns=['504'], prefix=['504'])


def random_forest_fp():
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X, Y)

    feature_importances = rf_classifier.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(5)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title(f'Random Forest Top 5 Feature Importance In {grade} For {score}')

    # Display the chart
    plt.tight_layout()
    plt.show()

    print(feature_importance_df)


def mean_accuracy_decrease():
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train your initial model
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train, Y_train)

    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(Y_test, baseline_model.predict(X_test))

    # Create an array to store accuracy decreases
    accuracy_decreases = []

    # Iterate through features and compute accuracy decreases
    for feature_name in X.columns:
        X_test_permuted = X_test.copy()
        X_test_permuted[feature_name] = np.random.permutation(X_test_permuted[feature_name])

        accuracy_permuted = accuracy_score(Y_test, baseline_model.predict(X_test_permuted))

        accuracy_decrease = baseline_accuracy - accuracy_permuted
        accuracy_decreases.append((feature_name, accuracy_decrease))

    accuracy_decreases.sort(key=lambda x: x[1], reverse=True)

    for feature_name, accuracy_decrease in accuracy_decreases:
        print(f"Feature: {feature_name}, Accuracy Decrease: {accuracy_decrease}")
    
    accuracy_decreases_df = pd.DataFrame(accuracy_decreases, columns=['Feature', 'Accuracy Decrease'])

    accuracy_decreases_df = accuracy_decreases_df.sort_values(by='Accuracy Decrease', ascending=False)

    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(accuracy_decreases_df['Feature'], accuracy_decreases_df['Accuracy Decrease'])
    plt.xlabel('Feature')
    plt.ylabel('Mean Accuracy Decrease')
    plt.title('Feature Importance: Mean Accuracy Decrease')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

    # Display the chart
    plt.tight_layout()
    plt.show()


def permutation():
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    rf_classifier.fit(X_train, Y_train)

    # Make predictions on the testing data
    Y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy: {accuracy}')

    # Generate classification report, confusion matrix, and ROC-AUC
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

    roc_auc_moderate = roc_auc_score((Y_test == 'Moderate').astype(int), (Y_pred == 'Moderate').astype(int))
    roc_auc_high = roc_auc_score((Y_test == 'High').astype(int), (Y_pred == 'High').astype(int))

    print(f'ROC AUC for "Moderate" class: {roc_auc_moderate}')
    print(f'ROC AUC for "High" class: {roc_auc_high}')


    feature_names = ['Unweighted GPA', 'Absence', 'Tardy', 'Dismissal', 'Endorse Q9', 'Gender Encoded']

    result = permutation_importance(
        rf_classifier, X_test, Y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    forest_importances = forest_importances.sort_values(ascending=False)


    # Create a bar plot of feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(forest_importances.index, forest_importances.values)
    plt.xlabel('Mean Permutation Importance')
    plt.title(f'Feature Importances for {score}')
    plt.show()

def feature_contribution_stratified_by_gender_race():
    features = ['Gender', 'Race / Ethnicity', 'ELL', 'SPED', '504', 
            'Unweighted GPA', 'Avg Level', 'Absence', 'Tardy', 'Dismissal', 'Endorse Q9', 'GAD-7 Total']
    features = ['Gender', 'Race / Ethnicity', 'ELL', 'SPED', '504', 
            'Unweighted GPA', 'Absence', 'Tardy', 'Dismissal', 'Endorse Q9']

    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(data[features].select_dtypes(include=['object']))
    encoded_feature_names = encoder.get_feature_names_out()

    numerical_features = data[features].select_dtypes(include=['int64', 'float64'])
    encoded_df = pd.concat([pd.DataFrame(encoded_features, columns=encoded_feature_names),
                            numerical_features.reset_index(drop=True)], axis=1)

    gender_groups = data['Gender'].unique()
    race_groups = data['Race / Ethnicity'].value_counts()
    race_groups = race_groups[race_groups > 10].index
    combined_groups = data.groupby(['Gender', 'Race / Ethnicity']).size()
    combined_groups = combined_groups[combined_groups > 10].index 


    rfc = RandomForestClassifier(random_state=42)

    stratified_kfold = StratifiedKFold(n_splits=2)

    scoring_metrics = {'accuracy': make_scorer(accuracy_score), 'f1': make_scorer(f1_score, average='weighted')}

    def compute_scores(mask, data, encoded_df, target):
        X_group = encoded_df[mask]
        y_group = data.loc[mask, target]
        scores = cross_validate(rfc, X_group, y_group, cv=stratified_kfold, scoring=scoring_metrics)
        return np.mean(scores['test_accuracy']), np.mean(scores['test_f1'])

    scores_gender = {g: compute_scores(data['Gender'] == g, data, encoded_df, target) for g in gender_groups}

    scores_race = {r: compute_scores(data['Race / Ethnicity'] == r, data, encoded_df, target) for r in race_groups}

    scores_combined = {}
    for (gender, race) in combined_groups:
        gender_mask = data['Gender'] == gender
        race_mask = data['Race / Ethnicity'] == race
        combined_mask = gender_mask & race_mask
        combined_key = f"{gender}_{race}"
        scores_combined[combined_key] = compute_scores(combined_mask, data, encoded_df, target)




    scores_gender = {g: compute_scores(data['Gender'] == g, data, encoded_df, target) for g in gender_groups}
    scores_race = {r: compute_scores(data['Race / Ethnicity'] == r, data, encoded_df, target) for r in race_groups}
    scores_combined = {f"{g}_{r}": compute_scores((data['Gender'] == g) & (data['Race / Ethnicity'] == r), data, encoded_df, target) 
                    for g, r in combined_groups}

    all_scores = {**scores_gender, **scores_race, **scores_combined}

    sorted_subgroups = sorted(all_scores.keys(), key=lambda x: (0 if x in gender_groups else 1 if x in race_groups else 2, x))

    sorted_accuracy_scores = [all_scores[sg][0] for sg in sorted_subgroups]
    sorted_f1_scores = [all_scores[sg][1] for sg in sorted_subgroups]

    x = np.arange(len(sorted_subgroups))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width/2, sorted_accuracy_scores, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, sorted_f1_scores, width, label='F1 Score')

    ax.set_ylabel('Scores')
    
    ax.set_title(f'Performance of RFC predictions of {target} in Groups for {grade}')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_subgroups, rotation=45, ha='right')
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    feature_contribution_stratified_by_gender_race()