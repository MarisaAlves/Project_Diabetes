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

How can bodily variables predict the progression of diabetes in patients.


### Abstract

4 sentence longer explanation about your research question. Include:

- opportunity (what data do we have)
To try and solve this question we have a datset of 442 diabetes patients who were measured on 10 baseline variables, as well as a measure of disease progression taken one year after baseline.
- challenge (what is the "problem" we could solve with this dataset)
The problem we are trying to solve with this problem is to form a predictive model which can be used on diabetic patients to assess based on their measurements whether they are at a higher risk for their disease to progress.
- action (how will we try to solve this problem/answer this question)
To try and solve this problem we will form a variety of machine learning algorithms based on our dataset which will be used to predict the measure of disease progression.
- resolution (what did we end up producing)
We ended up producing a linear regression equation which has the standardized baseline variables as variables for the model with corresponding coefficients.

### Introduction

Brief (no more than 1-2 paragraph) description about the dataset. Can copy from elsewhere, but cite the source (i.e. at least link, and explicitly say if it's copied from elsewhere).
This dataset contains the measurements of 442 diabetic patients. Measurements were made for these patients as follows: Age, sex, body mass index, blood pressure, map, tc, low-density lipoproteins, high-density lipoproteins, tch, and glu. Each of these 10 features have been mean centered and scaled by the standard deviation times n_samples. A year later, these same patients were measured for disease progression, and this is the target variable.



### Methods

Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
To solve the problem, I first began with cleaning up the dataset as needed. As the dataset as downloaded from sklearn is already standardized, not much was needed in this department, except as follows: outliers were assessed and removed as to further improve the model, and the sex variable was altered into two dummy variables as to not bias the data. Following this, a multiple linear regression model was conducted. While this model achieved good error, when examining the heatmap and pairplot of the variables, it is clear that not many of the variables seemed to correlate with diabetic disease progression. Thus, to improve the model, these uncorrelated variables should not be required in the regression equation. Therefore, in the end a Lasso regression analysis was performed, as it assumes only some of the variables will be important in the prediction. 

- why you chose this method

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
All of the links
http://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
https://scikit-learn.org/stable/datasets/index.html
-------
