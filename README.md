# Graduate Data Mining: Stroke Prediction and Inference
Project group only consisted of me [Mateo Forero]. The original project was done in R but a Python rendition was also created to show how each language can implement similar results. Aside from syntax, the biggest difference is that sklearn purposefully avoids statistical inference in models, so a different library was used for the logistic model. 

To see code output, refer to the [.MD file](Stroke_Data-Mining.md).

## Project Overview
This was a project for the graduate course Applied Data Mining and Analytics in Business. It uses the [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) found on Kaggle. The dataset was adjusted to only include adults (Age >= 18) because the risk factors associated with stroke in adolescents and children, such as genetic bleeding disorders, are not captured by this dataset.

The goal of the project was to execute data exploration and model creation in the hope of building a risk measure comparable to the physician established CHADS-VAS Score. This tool uses age, biological gender, presence of congestive heart failure, hypertension, history of strokes, heart disease and diabetes to predict risk of stroke. Creating a model that uses similar variables in addition to adding social determinants of health such as marriage status, work type, and residence type may provide a better prediction than is currently being used.

It's important to note that the overarching goal is not to predict strokes perfectly, but to adequately stratify patient risk.

### Models Used
- Binomial Logit Model with assumed dispersion coefficient of 1
- Gradient Boosting Classifier
- Random Forest Classifier

### Techniques Used
- Random Forest missing value imputations
- Grid Search hyperparameter optimization
- Logistic Model Step AIC parameter selection
- Cross Validation for model optimization and metric comparison
- Fold-only SMOTE to balance the dataset response proportions and prevent data leakage
- Custom threshold optimization cost function [F-Beta Score]
- Parallel computing to decrease model training time
- Data inference from model results
- Partial Dependence Plots
