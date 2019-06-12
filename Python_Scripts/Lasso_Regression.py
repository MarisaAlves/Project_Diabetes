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

diabetes = load_diabetes()

diabetes_df = pd.DataFrame(data=np.c_[diabetes.data, diabetes.target], columns=diabetes.feature_names + ['target'])

encoded_sex = pd.get_dummies(diabetes_df['sex'], drop_first=True)
diabetes_df = pd.concat([diabetes_df, encoded_sex], axis=1)
diabetes_df.rename(columns = {list(diabetes_df)[11]: "Encoded Sex"}, inplace=True)
diabetes_df.drop(['sex'], axis=1, inplace=True)

z = np.abs(stats.zscore(diabetes_df))

diabetes_df_o = diabetes_df[(z < 3).all(axis=1)]

X = diabetes_df_o.loc[:,['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'Encoded Sex']]
y = diabetes_df_o['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lasso = Lasso()
lasso.fit(X_train, y_train)

print(f"Intercept: {lasso.intercept_}\n")
print(f"Coeficients: {lasso.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lasso.coef_, X.columns)}")

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
plt.title('Ridge Residual Distribution')
plt.savefig('plots/Ridge_Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")