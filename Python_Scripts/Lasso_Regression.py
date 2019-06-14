from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats

os.makedirs('plots/', exist_ok=True)

#Loading the dataset from sklearn
diabetes = load_diabetes()

diabetes_df = pd.DataFrame(data=np.c_[diabetes.data, diabetes.target], columns=diabetes.feature_names + ['target'])
diabetes_df.columns = ['Age', 'Sex', 'BMI', 'BP', 'map', 'tc', 'ldl', 'hdl', 'tch', 'glu', 'Target']


#Exploratory analysis on dataset
diabetes_unaltered = pd.read_csv('diabetes.data',
                          sep='\s+',
                          header=0)
diabetes_unaltered.describe().to_csv("Diabetes_Describe.csv")

os.makedirs('plots', exist_ok=True)

sns.set(style='darkgrid', palette='coolwarm')

#Pairplot for differences between sexes
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

#Boxplots for distributions of features
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

#Lasso Regression analysis

#Dummy variable for sex feature
encoded_sex = pd.get_dummies(diabetes_df['Sex'], drop_first=True)
diabetes_df = pd.concat([diabetes_df, encoded_sex], axis=1)
diabetes_df.rename(columns = {list(diabetes_df)[11]: "Encoded Sex"}, inplace=True)
diabetes_df.drop(['Sex'], axis=1, inplace=True)

#Removing outliers based on Z Score
z = np.abs(stats.zscore(diabetes_df))
diabetes_df_o = diabetes_df[(z < 3).all(axis=1)]
X = diabetes_df_o.loc[:, ['Age', 'BMI', 'BP', 'map', 'tc', 'ldl', 'hdl', 'tch', 'glu', 'Encoded Sex']]
y = diabetes_df_o['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lasso = Lasso()
lasso.fit(X_train, y_train)

print(f"Intercept: {lasso.intercept_}\n")
print(f"Coeficients: {lasso.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lasso.coef_, X.columns)}")
pd.DataFrame(lasso.coef_, X.columns).to_csv("Lasso Coefficients")

predicted_values = lasso.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

sns.set(palette="hls")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="+")
plt.plot([0, 300], [0, 300], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Lasso Real Value vs Predicted Values')
plt.savefig('plots/Lasso_Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker="s")
plt.plot([200, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Lasso Real Value vs Residuals')
plt.savefig('plots/Lasso_Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Lasso Residual Distribution')
plt.savefig('plots/Lasso_Residual_Distn.png')
plt.clf()
plt.close()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")