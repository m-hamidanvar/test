# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:47:48 2024

@author: Mary
"""
#

import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#%%
df = pd.read_csv('D:\Learning\ML\Datasets\diabetes_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())

#%%Balancing and Cleaning the Dataset
d = sns.countplot(data = df, x = "diabetes")
d.set_title("Number of Patients With and Without Diabetes")
d.set_xlabel("Diagnosis")
d.set_xticklabels(["Non-Diabetic", "Diabetic"])
plt.show()

#%%
df_race = df[["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"]]
by_race = pd.from_dummies(df_race)
df2 = df.drop(columns = ["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"])
df2.insert(2, "race", by_race)
df2["race"] = df2["race"].str.replace("race:", "")
df2["race"] = df2["race"].str.replace("AfricanAmerican", "African-American")

#%%
bins = [0, 18, 25, 35, 45, 55, 65, np.inf]
age_order = ["0-18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df2.insert(3, "age_cat", pd.cut(df2["age"], bins, labels = age_order))

#%%
bins = [0, 18.5, 25, 30, 35, 40, np.inf]
bmi_order = ["Underweight", "Normal", "Overweight", "Moderately Obese", "Severely Obese", "Morbidly Obese"]
df2.insert(9, "bmi_cat", pd.cut(df2["bmi"], bins, labels = bmi_order))

#%%
print("before:",df.shape)
df2["smoking_history"] = df2["smoking_history"].str.title()
df2 = df2[df2["smoking_history"] != "No Info"]
print("after:",df2.shape)

#%%
columns = ['gender', 'race', 'age_cat', 'location', 'hypertension','heart_disease', 'smoking_history', 'bmi_cat', 'diabetes']

for col in columns:
    
    if col in ['gender', 'race', 'age_cat', 'location', 'smoking_history', 'bmi_cat']:
        df2[col] = df2[col].astype("category")
        
    else:
        df2[col] = df2[col].astype("bool")

#%%
columns = list(df2.columns)
new_name = [col.title() for col in columns]



for c, n in zip(columns, new_name):
    df2 = df2.rename(columns = {c:n})

df2 = df2.rename(columns = {"Age_Cat":"Age Group",
                            "Heart_Disease":"Heart Disease", 
                            "Smoking_History":"Smoking History",
                            "Bmi_Cat":"BMI Category",
                            "Bmi":"BMI",
                            "Hba1C_Level":"HbA1C Level",
                            "Blood_Glucose_Level":"Blood Glucose Level"
                           }
                )
print(df2.head())

#%%
diabetes_yes = df2[df2["Diabetes"] == True]
diabetes_no = df2[df2["Diabetes"] == False].sample(n = diabetes_yes.shape[0], ignore_index = True, random_state = 42)

sample = pd.concat([diabetes_yes, diabetes_no], axis = 0).reset_index(drop = True)

print(sample.head())

#%%
d = sns.countplot(data = sample, x = "Diabetes")
d.set_title("Number of Patients With and Without Diabetes")
d.set_xlabel("Diagnosis")
d.set_xticklabels(["Non-Diabetic", "Diabetic"])
plt.show()

#%%Exploratory Data Analysis
cat_columns = ["Gender",
               "Location",
               "Race", 
               "Age Group", 
               "Hypertension", 
               "Heart Disease", 
               "Smoking History", 
               "BMI Category"]

for col in cat_columns:
    
    categories = list(sample[col].unique())
    
    non_diabetic_counts = []
    diabetic_counts = []
    
    for cat in categories:
        
        non_diabetic = sample[(sample[col] == cat) & (sample["Diabetes"] == False)].shape[0]
        diabetic = sample[(sample[col] == cat) & (sample["Diabetes"] == True)].shape[0]
        
        non_diabetic_counts.append(non_diabetic)
        diabetic_counts.append(diabetic)
    
    d = pd.DataFrame({col: categories, "Non-diabetic": non_diabetic_counts, "Diabetic": diabetic_counts})
    
    if col == "Age Group":
        d[col] = pd.Categorical(d[col], categories = age_order, ordered=True)
        d_sorted = d.sort_values(col)
        d_sorted.plot(x = col, kind = "bar", stacked = True)
           
    if col == "BMI Category":
        d[col] = pd.Categorical(d[col], categories = bmi_order, ordered=True)
        d_sorted = d.sort_values(col)
        d_sorted.plot(x = col, kind = "bar", stacked = True)        
    
    else:
        d.plot(x = col, kind = "bar", stacked = True)
            
    plt.title(f"Patients According to {col} and Diabetes Status")
    plt.xlabel(f"{col}")
    plt.ylabel("Count")
    plt.legend(title = "Diabetes Status")
    plt.show()
    
#%%
num_columns = ["Age", "BMI", "HbA1C Level", "Blood Glucose Level"]

for col in num_columns:
    n = sns.boxplot(data = sample, x = "Diabetes", y = col)
    n.set_title(f"{col} of Patients")
    plt.show()
    
#%%Correlation Matrix
df_encoded = sample.drop(columns = ["Age Group", "BMI Category"])
le = LabelEncoder()

df_encoded["Gender"] = le.fit_transform(df_encoded["Gender"])
df_encoded["Location"] = le.fit_transform(df_encoded["Location"])
df_encoded["Race"] = le.fit_transform(df_encoded["Race"])
df_encoded["Smoking History"] = le.fit_transform(df_encoded["Smoking History"])
df_encoded

#%%
corr_mat = df_encoded.corr()
corr_mat

plt.figure(figsize=(10, 10))
sns.heatmap(corr_mat, annot = True, cmap = "coolwarm", fmt='.2f')
plt.title("Correlation Matrix of Diabetes Dataset")
plt.show()

#%%Drill-down Analysis
columns =  ["Blood Glucose Level", "HbA1C Level"]

for col in columns:
    
    c = sns.boxplot(data = sample, x = "Diabetes", y = col)
    c.set_title(f"{col} of Patients")
    c.set_xlabel("Diabetes Status")
    c.set_ylabel(f"col")
    c.set_xticklabels(["Non-Diabetic", "Diabetic"])
    
    if col == "Blood Glucose Level":
        c.axhline(y = 200, c = "red", ls = "--", label = "Diabetic Sugar Levels")
        
    else:
        c.axhline(y = 5.7, c = "yellow", ls = "--",label = "pre-diabetes")
        c.axhline(y = 6.4, c = "red", ls = "--",label = "diabetes")
        
    c.legend()
    plt.show()
    
#%%
columns =  ["Blood Glucose Level", "HbA1C Level"]

for col in columns:
    
    Nondiabetic = sample[sample["Diabetes"] == 0][col]
    Diabetic = sample[sample["Diabetes"] == 1][col]
    data = [Nondiabetic, Diabetic]
    
    for d in data:
        stats, pval = shapiro(d)
        
        if pval <= 0.05:
            print(f"The distribution of {col} is not normal.")
            
        else:
            print(f"The distribution of {col} is normal.")
#%%mannwhitneyu
columns =  ["Blood Glucose Level", "HbA1C Level"]

for col in columns:
    
    stat, pval = mannwhitneyu(
        sample[sample["Diabetes"] == 1][col], 
        sample[sample["Diabetes"] == 0][col], alternative = "greater")
    
    if pval > 0.05:
        print(f"p-value = {pval}. The {col} of patients with diabetes are not higher than those with diabetes.")
    
    else:
        print(f"p-value = {pval}. The {col} of patients with diabetes are higher than those with diabetes.")
            
#%%
targets = ["Blood Glucose Level", "HbA1C Level"]
cat_columns.remove("Location")

for col in cat_columns:
    
    for target in targets:
        h = sns.boxplot(data = sample, x = col, y = target, hue = "Diabetes")
        h.set_title(f"{target} Among Patients According to {col}")
        h.set_xlabel(f"{col}")
        h.set_ylabel(f"{target}")
        
        if target == "Blood Glucose Level":            
            h.axhline(y = 200, c = "red", ls = "--",label = "Diabetes")
            
        else:            
            h.axhline(y = 5.7, c = "yellow", ls = "--",label = "Pre-diabetes")
            h.axhline(y = 6.4, c = "red", ls = "--",label = "Diabetes")
            
        h.legend()
        plt.show()

#%%
columns = ["Age", "BMI"]
targets = ["Blood Glucose Level", "HbA1C Level"]

for col in columns:
    
    for target in targets:
        
        h = sns.scatterplot(data = sample, x = col, y = target, hue = "Diabetes") 
        h.set_title(f"{target} Among Patients According to {col}")
        
        if col == "BMI":
            h.axvline(x = 30, c = "green", ls = "--",label = "Obese")
        
        h.set_xlabel(f"{col}")
        h.set_ylabel(f"{target}")
        
        if target == "Blood Glucose Level":
            h.axhline(y = 200, c = "red", ls = "--",label = "Diabetes")
            
        else:
            h.axhline(y = 5.7, c = "yellow", ls = "--",label = "Pre-diabetes")
            h.axhline(y = 6.4, c = "red", ls = "--",label = "Diabetes")
            
        h.legend()
        plt.show()
        
#%%Statistical Tests for Relationship/Independence
cat_columns.append("Location")

for col in cat_columns:
    contingency_table  = pd.crosstab(sample[col], sample["Diabetes"]) 
    print(col,"contingency_table::::")
    print()
    print(contingency_table,"********")
    chi_square, p, dof, expected = chi2_contingency(contingency_table)
    print("dof:",dof,"...expexted:",expected)
    critical_value = chi2.ppf(1 - 0.05, dof)
                                     
    if chi_square <= critical_value:
        print(f"There is no relationship between {col} and diabetes.")
        cat_columns.remove(col)
                                     
    else:
        print(f"There is a relationship between {col} and diabetes.")
        
#%%
for col in cat_columns:
    
    contingency_table  = pd.crosstab(sample[col], sample["Diabetes"])
    contingency_table["total"] = contingency_table[False] + contingency_table[True]
    contingency_table["risk"] = contingency_table[True] / contingency_table["total"]
    risk_A_label = contingency_table["risk"].idxmax()
    risk_A = contingency_table.loc[risk_A_label, "risk"]
    risk_B = contingency_table.loc[contingency_table.index != risk_A_label, "risk"].mean()
    rr = round(risk_A/risk_B, 2)
    
    sns.heatmap(contingency_table.drop(columns = [True, False, "total"]), vmin = 0, vmax = 1, annot = True, cmap = "Reds", fmt = ".4f")
    plt.title(f"Risk of Diabetes According to {col}")
    plt.show()
    
    if col == "Age Group":        
        print(f"Ages {risk_A_label}: {rr} times more likely to develop diabetes.")
        
    elif col in ["Hypertension", "Heart Disease"]:
        if risk_A_label == True:
            print(f"People with {col}: {rr} times more likely to develop diabetes.")
        else:
            print(f"People without {col}: {rr} times more likely to develop diabetes.")
            
    elif col == "Smoking History":
        print(f"{risk_A_label} smokers: {rr} times more likely to develop diabetes.")
    
    elif col == "BMI Category":
        print(f"{risk_A_label} people: {rr} times more likely to develop diabetes.")
        
    else:        
        print(f"{risk_A_label}: {rr} times more likely to develop diabetes.")






