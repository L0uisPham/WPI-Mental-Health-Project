import pandas as pd
import pingouin as pg
from IPython.display import display
import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="covariance of constraints does not have full rank")


# Function to load your dataset for 7th grade
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# ANOVA function for Grade 7 and Grade 8
def run_ANOVA_grade_7_8(file_path):
    df_grade_7 = load_data(file_path)
    
    # You can specify other question names here if needed
    q_names = ['Total Depression Score', 'Depression T Score', 'Total Anxiety Score', 'Anxiety T Score', 'Endorse Q9']
    anova_tables = []
    p_unc_list = []
    for q in q_names:
        # Running the ANOVA
        model1 = pg.anova(dv=q, between=['Gender', 'Race / Ethnicity'], data=df_grade_7, detailed=True)
        display(model1)
        anova_tables.append(model1)
        col = model1['p-unc']
        col = col.iloc[0:3]
        col.name = q

        p_unc_list.append(col)

    # Combining the p-values into a DataFrame
    df_tabbed = pd.concat(p_unc_list, axis=1)
    df_tabbed = df_tabbed.rename(index={0: 'Gender (ME)', 1: 'Race / Ethnicity (ME)', 2: 'Gender*Race / Ethnicity (IE)'})

    # Formatting the p-values for display
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
    display(df_tabbed)
    df_tabbed.to_csv('formatted_p_values.csv', index=True)

    # Return the ANOVA tables if needed for further processing or exporting
    return anova_tables


# ANOVA function for Grade 7
def run_ANOVA_grade_9_11(file_path):
    df_grade_7 = load_data(file_path)
    
    # You can specify other question names here if needed
    q_names = ['PHQ-9 Total', 'GAD-7 Total', 'Endorse Q9']

    anova_tables = []
    p_unc_list = []
    for q in q_names:
        # Running the ANOVA
        model1 = pg.anova(dv=q, between=['Gender', 'Race / Ethnicity'], data=df_grade_7, detailed=True)
        display(model1)
        anova_tables.append(model1)
        col = model1['p-unc']
        col = col.iloc[0:3]
        col.name = q

        p_unc_list.append(col)

    # Combining the p-values into a DataFrame
    df_tabbed = pd.concat(p_unc_list, axis=1)
    df_tabbed = df_tabbed.rename(index={0: 'Gender (ME)', 1: 'Race / Ethnicity (ME)', 2: 'Gender*Race / Ethnicity (IE)'})

    # Formatting the p-values for display
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
    display(df_tabbed)

    df_tabbed.to_csv('formatted_p_values.csv', index=True)

    # Return the ANOVA tables if needed for further processing or exporting
    return anova_tables

file_path = 'new_data/Grade 11.csv'  

anova_results = run_ANOVA_grade_9_11(file_path)
print(anova_results)
for i, table in enumerate(anova_results):
    table.to_csv(f'anova_result_{i}.csv', index=False)
#run_ANOVA_grade_9_11(file_path)