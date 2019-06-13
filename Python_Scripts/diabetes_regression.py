from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from scipy import stats

diabetes = load_diabetes()

diabetes_df = pd.DataFrame(data=np.c_[diabetes.data, diabetes.target], columns=diabetes.feature_names + ['target'])
diabetes_df.columns = ['Age', 'Sex', 'BMI', 'BP', 'map', 'tc', 'ldl', 'hdl', 'tch', 'glu', 'Target']

encoded_sex = pd.get_dummies(diabetes_df['Sex'], drop_first=True)
diabetes_df = pd.concat([diabetes_df, encoded_sex], axis=1)
diabetes_df.rename(columns = {list(diabetes_df)[11]: "Encoded Sex"}, inplace=True)
diabetes_df.drop(['Sex'], axis=1, inplace=True)

z = np.abs(stats.zscore(diabetes_df))

diabetes_df_o = diabetes_df[(z < 3).all(axis=1)]

print(diabetes_df.shape)
print(diabetes_df_o.shape)

X = diabetes_df_o.loc[:, ['Age', 'BMI', 'BP', 'map', 'tc', 'ldl', 'hdl', 'tch', 'glu', 'Encoded Sex']]
y = diabetes_df_o['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, X.columns)}")
pd.DataFrame(lm.coef_, X.columns).to_csv("Linear Regression Coefficients")

predicted_values = lm.predict(X_test)

os.makedirs('plots/', exist_ok=True)

sns.set(palette="Paired")
residuals = y_test - predicted_values

sns.scatterplot(y_test, predicted_values, marker="H")
plt.plot([0, 300], [0, 300], ':', linewidth=2.0, color='g')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('Linear Regression Real Value vs Predicted Values')
plt.savefig('plots/Linear_Predicted.png')
plt.clf()

sns.scatterplot(y_test, residuals, marker=5)
plt.plot([300, 0], [0, 0], ':',linewidth=2.0, color='g')
plt.xlabel('Real Value')
plt.ylabel('Residuals')
plt.title('Linear Regression Real Value vs Residuals')
plt.savefig('plots/Linear_Residuals.png')
plt.clf()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], ':', linewidth=2.0, )
plt.title('Linear Regression Residual Distribution', color='g')
plt.savefig('plots/Linear_Residual_Distn.png')
plt.clf()

print(f"MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
