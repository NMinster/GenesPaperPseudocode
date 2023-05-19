# GenesPaperPseudocode
Paper title: "A Machine Learning Approach to Parkinson’s Disease Blood Transcriptomics"

This is a translation of the following pseudocode provided in the paper (https://www.mdpi.com/2073-4425/13/5/727).

This code performs a complex iterative feature selection process using RandomForest and XGBoost models. The process could be summarized as follows:
1.	Let F be the total number of features in the dataset.
2.	The code then starts a 20-round process (r from 1 to 20). For each round:
  a. It divides the dataset into 10 stratified folds using a different random seed each round.
  b. For each fold (total 10 folds), it sets one fold as validation data and the remaining 9 as training data.
  c. It then initiates another 100-round process (s from 1 to 100), where it:
    •	Divides the training data into 5 stratified folds using a new random seed each round.
    •	Takes 4 of these folds as a new training set and trains a RandomForest with 1000 trees on it.
    •	It calculates the feature importance for each feature and identifies if the feature is an outlier based on a threshold derived from the median and interquartile range (IQR) of importance.
    •	It also calculates the percentage of times each feature was identified as an outlier in the 100 rounds.
  d. For a range of C values (1 to 100), it selects features that were identified as outliers more than C times and trains an XGBoost model on these features. The model's performance (ROC AUC) is evaluated on the validation set.
3.	After all rounds and folds, it calculates the median performance across all folds (m_ROCAUCr,C) and rounds (m_ROCAUCC) for each C.
4.	It then finds the C* that gives the maximum median performance (m_ROCAUCC).
5.	Finally, for each feature, it calculates the number of times it was selected (count_selectedf) in all 20 rounds at C*.
The main goal of the script is to identify the optimal subset of features and the threshold (C*) that maximize the performance (ROC AUC) of an XGBoost model. It does this through extensive feature importance estimation and outlier detection via RandomForest, followed by model training and performance evaluation via XGBoost. It also incorporates multiple iterations (20 rounds and 100 sub-rounds) and cross-validation (10-fold and 5-fold) to ensure robustness and generalizability of the results.

Here are a few suggestions for improving the script:

1.  Feature Scaling: Gene expression data often contains large outliers and variance among its features. You might want to apply some form of feature scaling, such as StandardScaler or MinMaxScaler from the sklearn.preprocessing module, before training your models.

2.  Feature Importance Interpretation: Random Forest and XGBoost both provide feature importances, which can be interpreted as the contribution of each feature to the model's predictions. You can use these to better understand which genes (features) are the most informative for your classification task. You might want to visualize these feature importances for better interpretation.

3.  Hyperparameter Tuning: The performance of both RandomForest and XGBoost models can be significantly affected by their hyperparameters. Consider using a tool like GridSearchCV or RandomizedSearchCV from sklearn.model_selection to find the best hyperparameters for your models.

4.  Model Interpretation: Consider using a tool like SHAP (SHapley Additive exPlanations) for a more in-depth interpretation of your models. SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

5.  Multiple Models: Consider training several different types of models and compare their performance. Ensemble models, support vector machines, or deep learning models may yield better results.

6.  Balancing the Classes: If your binary classification task has imbalanced classes (one class has significantly more samples than the other), this can bias your model's predictions. Consider using a technique like SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN (Adaptive Synthetic Sampling) to balance your classes.

Here are a few suggestions for improving the methodology:

1.  This data is grouped by patient and visit. Many patients visit many times over several years. The implementation in the paper does not consider this and patient samples are mixed among training, validation, and test sets. This will likely lead to over fitting. You should consider this when splitting the data. 

2.  More care needs to be taken to validate machine learning results. The models need to be tested on unseen data, external to PPMI.

4.  Finally, these procedures need to be articulated more clearly and in a manner that can be reproduced by other researchers.

