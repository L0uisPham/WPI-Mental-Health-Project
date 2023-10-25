import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_excel("d2022.xlsx") #change this into the file name.

df_cleaned = df.dropna()

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

df_cleaned['GAD 7 Category'] = df_cleaned['GAD 7 Risk'].apply(categorize_gad7_risk)

# remove some for covariates, like, you can leave some test there if needed.
df_tsne = df_cleaned.drop(columns=['GAD 7 Category', 'PHQ 9 Risk', 'ID MAPPER','Endorse Q9','GAD7-1','GAD7-2','GAD7-3','GAD7-4','GAD7-5','GAD7-6','GAD7-7','GAD 7 Risk','PHQ9-1','PHQ9-2','PHQ9-3','PHQ9-4','PHQ9-5','PHQ9-6','PHQ9-7','PHQ9-8','PHQ9-9'])

df_encoded = pd.get_dummies(df_tsne.select_dtypes(include=['object']))
df_numeric = df_tsne.select_dtypes(exclude=['object'])
df_preprocessed = pd.concat([df_numeric, df_encoded], axis=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_preprocessed)
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(df_scaled)
color_map = {
    "none to minimal": "green",
    "mild": "yellowgreen",
    "moderate": "yellow",
    "severe": "red"
}
color_map_phq9 = {
    "none to minimal": "green",
    "mild": "yellowgreen",
    "moderate": "yellow",
    "moderately severe": "orange",
    "severe": "red"
}

plt.figure(figsize=(12, 8))
for category, color in color_map.items():
    subset = tsne_results[df_cleaned['GAD 7 Category'] == category]
    plt.scatter(subset[:, 0], subset[:, 1], color=color, label=category, alpha=0.6)

plt.title('t-SNE Visualization Colored by GAD-7 Risk Categories')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#covariates change the plot drastically so play around with it.