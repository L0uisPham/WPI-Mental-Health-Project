# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset (you've done this step before)
df = pd.read_excel("d2022.xlsx")

# Drop rows with missing values
df_cleaned = df.dropna()

# Prepare the data
# Define a function to categorize GAD 7 Risk into the specified groups
def categorize_gad7_risk(value):
    if 0 <= value <= 4:
        return "none to minimal"
    elif 5 <= value <= 9:
        return "mild"
    elif 10 <= value <= 14:
        return "moderate"
    elif 15 <= value <= 21:
        return "severe"
def categorize_phq9_risk(value):
    if 0 <= value <= 4:
        return "none to minimal"
    elif 5 <= value <= 9:
        return "mild"
    elif 10 <= value <= 14:
        return "moderate"
    elif 15 <= value <= 19:
        return "moderately severe"
    elif 20 <= value <= 27:
        return "severe"
# Separate the target variable (PHQ 9 Risk categories) and the features
X = df_cleaned.drop(columns=['PHQ 9 Risk', 'ID MAPPER','Endorse Q9','GAD7-1','GAD7-2','GAD7-3','GAD7-4','GAD7-5','GAD7-6','GAD7-7','GAD 7 Risk','PHQ9-1','PHQ9-2','PHQ9-3','PHQ9-4','PHQ9-5','PHQ9-6','PHQ9-7','PHQ9-8','PHQ9-9'])
y = df_cleaned['PHQ 9 Risk'].apply(categorize_phq9_risk)  # Using the categorize_phq9_risk function from before

# Convert categorical columns to numerical using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)



from sklearn.model_selection import cross_val_score

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k-fold cross-validation (let's use k=5 as an example)
scores = cross_val_score(clf, X_encoded, y, cv=10)

average_score = scores.mean()
print("Average Accuracy:", average_score)
print("Scores:", scores)

# Fit the Random Forest model to the data
clf.fit(X_encoded, y)

# Select a random sample from the dataset
sample = X_encoded.sample(1)
prediction = clf.predict(sample)

# Extract decision path from one of the trees (for demonstration purposes, we'll use the first tree)
tree = clf.estimators_[0]
path = tree.decision_path(sample)

# Get the decision rules
feature_names = X_encoded.columns
rules = []
for node_id in path.indices:
    if tree.tree_.feature[node_id] != -2:
        feature_name = feature_names[tree.tree_.feature[node_id]]
        threshold = tree.tree_.threshold[node_id]
        if sample[feature_name].values[0] <= threshold:
            rule = "{} <= {}".format(feature_name, threshold)
        else:
            rule = "{} > {}".format(feature_name, threshold)
        rules.append(rule)

# Print the prediction and the decision rules
print("Prediction:", prediction[0])
print("Decision Rules:")
for rule in rules:
    print(rule)

