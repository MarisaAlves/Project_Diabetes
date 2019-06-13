import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
feature_names = diabetes.feature_names

X = diabetes.data
Y = diabetes.target

diabetes_df = pd.DataFrame(data=np.c_[diabetes.data, diabetes.target], columns=diabetes.feature_names + ['target'])

diabetes_df.columns = ['Age', 'Sex', 'BMI', 'BP', 'map', 'tc', 'ldl', 'hdl', 'tch', 'glu', 'Target']

diabetes_unaltered = pd.read_csv('diabetes.data',
                          sep='\s+',
                          header=0)
diabetes_unaltered.describe().to_csv("Diabetes_Describe")

os.makedirs('plots', exist_ok=True)

sns.set(style='darkgrid', palette='coolwarm')

#Pairplot
sns.pairplot(diabetes_df, hue='Sex', diag_kind='hist')
plt.savefig('plots/diabetes_pairplot.png')
plt.clf()

#Heatmap
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(diabetes_df.corr(), annot=True, cmap='RdYlGn')
ax.set_xticklabels(diabetes_df.columns, rotation=45)
ax.set_yticklabels(diabetes_df.columns, rotation=45)
plt.title('Heatmap for Diabetes Variables')
plt.savefig('plots/diabetes_heatmap.png')

plt.close()

#Boxplots
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(10, 5))
for indx, col in enumerate(diabetes_unaltered.columns):
    outliers = dict(markerfacecolor='r', marker='D')
    if col != 'Sex':
        if indx == 0:
            axes[0][indx].boxplot(diabetes_unaltered[col], flierprops=outliers)
            axes[0][indx].set_ylabel(col)
        elif indx <= 5 and indx >=2:
            axes[0][6-indx].boxplot(diabetes_unaltered[col], flierprops=outliers)
            axes[0][6-indx].set_ylabel(col)
        elif indx <=10 and indx>=6:
            axes[1][10-indx].boxplot(diabetes_unaltered[col], flierprops=outliers)
            axes[1][10-indx].set_ylabel(col)

fig.suptitle("Boxplots for Diabetes Features")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(f'plots/Boxplots_features')
plt.clf()

#Scatter Plot
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.grid(axis='y', alpha=0.5)
axes.scatter(diabetes_df["tch"], diabetes_df["Target"], marker="1", color='blue')
axes.scatter(diabetes_df["BP"], diabetes_df["Target"], marker="*", color='orange')
axes.scatter(diabetes_df["BMI"], diabetes_df["Target"], marker=".", color='green')
axes.set_title(f'Diabetes comparisons')
axes.set_ylabel('Diabetes Progression Indicator')
axes.set_xlabel('Feature Levels')
axes.legend(("tch", "BP", "BMI"))
plt.savefig(f'plots/diabetesProgression_to_tch_BP_BMI.png', dpi=300)
plt.clf()
plt.close()