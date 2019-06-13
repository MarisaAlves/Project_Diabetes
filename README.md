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

How can individual factors and bodily variables predict the progression of diabetes in prediagnosed patients?

### Abstract

<p> To try and solve this question we have a dataset of 442 diabetic patients who were measured on 10 baseline variables, as well as a measure of disease progression taken one year after baseline.
The challenge we are trying to solve with this problem is to form a predictive model which can be used on diabetic patients to assess based on their measurements whether they are at a higher risk for their disease to progress.
To try and solve this problem we will form a variety of machine learning algorithms based on our dataset which will be used to predict the measure of disease progression. We ended up producing a linear regression equation which has the standardized baseline variables as variables for the model with corresponding coefficients. <p>

### Introduction

<p> Diabetes has long been a disease chronically associated with patients who are overweight or obese. But what other factors impact diabetic disease progression? To attempt to answer this question we have a dataset containing the measurements of 442 diabetic patients. Measurements were made for these patients as follows: Age, sex, body mass index(bmi), blood pressure(bp), map, tc, low-density lipoproteins(ldl), high-density lipoproteins(hdl), total cholesterol(tch), and glu. Each of these 10 features have been mean centered and scaled by the standard deviation times n_samples. A year later, these same patients were measured for disease progression, and this is the target variable, the variable we are seeking to predict. <p>


### Methods

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

<p> To solve the problem, I first began with cleaning up the dataset as needed. As the dataset was loaded into python from sklearn already standardized, not much was needed in this department, except the following two methods. First, outliers were assessed and removed as to further improve the model. The decision to find and remove these observations that lie far outside the normal range was largely based on the boxplots for each variable (see Boxplots_features.png). For these, the outliers were colored in red, and we can see there are quite a few present. Therefore, to enhance the model 12 observations were removed from the analysis. Lastly, the sex variable was altered into two dummy variables (one of which was removed).This was performed as to not bias the data into forming a model based on the thinking that males are "superior" based on their value of 1. <p>
<p> Following the data clean up, a multiple linear regression model was conducted. While this model achieved good error, when examining the heatmap and pairplot of the variables, it is clear that not many of the variables seemed to correlate with diabetic disease progression (see diabetes_pairplot.png and diabetes_heatmap.png). Thus, to improve the model, these uncorrelated variables should not be required in the regression equation. Therefore, in the end a Lasso regression analysis was performed. Lasso is a technique to reduce model complexity and prevent over-fitting as it prefers a solution with fewer coefficients. This analysis was achieved using the Lasso Regressor which is built into scikit-learn. <p>

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
Figures to include:
Scatter plot for BP, BMI and S5
Lasso Predicted

We see from the above scatter plot there is a correlation between the blood pressure, bmi, and total cholesterol: as they increase, so too does the diabetic disease progressor variable.

<p> In the lasso regression model, we achieved a regression equation as follows: Target = 308.078485*BMI + 48.349328*BP + 270.965511*tch. This model achieved and root mean square error (RMSE) value of 57.77373, which is the standard deviation of the residuals, ie, the dispersion of the differences between the predicted values and the test values. <p>


### Discussion

<p> Overall, the final model was satisfactory. It achieved a good level of error and could decently predict the diabetic disease progression for the test set. However, there is still room for improvement. To start, enhance this model I would gather more data. Prior to analysis, this dataset contains the values of 442 diabetic patients. After removal of outlier values, it contained only 430. This is considered to be a small sample size, especially when you consider the fact that there are hundreds of millions of diabetic patients around the world. Therefore, our model is limited in its validity because the more samples, the higher precision of the regression model. After more data has been collected, it may even be possible that a different test would be more likely, and thus the methods for analysis may need to be altered. <p>

### References

All of the links

http://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
https://scikit-learn.org/stable/datasets/index.html
-------
