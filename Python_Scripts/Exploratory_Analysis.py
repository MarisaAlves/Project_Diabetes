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

diabetes_unaltered = pd.read_csv('diabetes.data',
                          sep='\s+',
                          header=0)
print(diabetes_unaltered.describe().to_string())

os.makedirs('plots', exist_ok=True)

sns.set(style='darkgrid', palette='coolwarm')

#Pairplot
sns.pairplot(diabetes_df, hue='sex', diag_kind='hist')
plt.savefig('plots/diabetes_pairplot.png')
plt.clf()

#Heatmap
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(diabetes_df.corr(), annot=True, cmap='autumn')
ax.set_xticklabels(diabetes_df.columns, rotation=45)
ax.set_yticklabels(diabetes_df.columns, rotation=45)
plt.savefig('plots/diabetes_heatmap.png')

plt.close()