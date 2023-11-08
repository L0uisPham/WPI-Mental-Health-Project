import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load dataset
grade = 'Grade 7' 
data = pd.read_csv(f'new_data/{grade}.csv')
moderate_threshold = 40
high_threshold = 70
mode = {
    0: 'Depression T Score',
    1: 'Anxiety T Score',
    2: 'PHQ-9 Total',
    3: 'GAD-7 Total ',
    4: 'Total T Score'
}

score = mode[1]


# Encode categorical variables 
le = LabelEncoder()


X = data[['Unweighted GPA', 'Endorse Q9', 'Absence', 'Tardy', 'Dismissal', 'Gender', 'Race / Ethnicity', 'ELL', 'SPED', '504']]
data['target_column'] = pd.cut(data[score], bins=[0, moderate_threshold, high_threshold, float('inf')],
                               labels=['Low', 'Moderate', 'High'], right=False)

Y = data['target_column']

X = pd.get_dummies(X, columns=['Gender'], prefix=['Gender'])

X = pd.get_dummies(X, columns=['Race / Ethnicity'], prefix=['Race'])

X = pd.get_dummies(X, columns=['SPED'], prefix=['SPED'])

X = pd.get_dummies(X, columns=['ELL'], prefix=['ELL'])

X = pd.get_dummies(X, columns=['504'], prefix=['504'])

data['target_column'] = pd.cut(data[score], bins=[0, moderate_threshold, high_threshold, float('inf')],
                               labels=['Low', 'Moderate', 'High'], right=False)
Y = data['target_column']


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


if __name__ == '__main__':
    random_forest_fp()