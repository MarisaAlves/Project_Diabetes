Diabetes Project

| Name | Date |
|:-------|:---------------|
|Marisa Alves | June |

-----

### Resources
Your repository should include the following:

- Python script for your analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

How can bodily variables predict the progression of diabetes in diabetic patients.


### Abstract

To try and solve this question we have a datset of 442 diabetes patients who were measured on 10 baseline variables, as well as a measure of disease progression taken one year after baseline.
The challenge we are trying to solve with this problem is to form a predictive model which can be used on diabetic patients to assess based on their measurements whether they are at a higher risk for their disease to progress.
To try and solve this problem we will form a variety of machine learning algorithms based on our dataset which will be used to predict the measure of disease progression.
We ended up producing a linear regression equation which has the standardized baseline variables as variables for the model with corresponding coefficients.

### Introduction

Diabetes has long been a disease chronically associated with patients who are overweight or obese. But what other factors impact diabetic disease progression? To try and answer this question we have a dataset containing the measurements of 442 diabetic patients. Measurements were made for these patients as follows: Age, sex, body mass index, blood pressure, map, tc, low-density lipoproteins, high-density lipoproteins, tch, and glu. Each of these 10 features have been mean centered and scaled by the standard deviation times n_samples. A year later, these same patients were measured for disease progression, and this is the target variable.


### Methods

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

To solve the problem, I first began with cleaning up the dataset as needed. As the dataset as downloaded from sklearn is already standardized, not much was needed in this department, except as follows: outliers were assessed and removed as to further improve the model, and the sex variable was altered into two dummy variables (one of which was removed) as to not bias the data. Following this, a multiple linear regression model was conducted. While this model achieved good error, when examining the heatmap and pairplot of the variables, it is clear that not many of the variables seemed to correlate with diabetic disease progression. Thus, to improve the model, these uncorrelated variables should not be required in the regression equation. Therefore, in the end a Lasso regression analysis was performed, as it assumes only some of the variables will be important in the prediction. 

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
Figures to include:
Scatter plot for BP, BMI and S5
Lasso Predicted

- At least 1 "value" that summarizes either your data or the "performance" of your method
Include regression equation and RMSE value

- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem

- suggested next step that could make it better.
To improve on this model I would gather more data. Prior to analysis, this dataset contains the values of 442 diabetic patients. After removal of outlier values, it contained only 430. This is considered to be a small sample size, especially when you consider the fact that there are ____ diabetic patients around the world. Therefore, our model is limited in its validity because the more samples, the higher precision of the regression model.

### References

All of the links

http://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
https://scikit-learn.org/stable/datasets/index.html
-------
