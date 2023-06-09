#This is a similar method used by the paper with improvements

#Improvements include modularization, a proper grid search, multiple models, proper class imbalance handeling, and use of groupkfold to control for multiple samples from the same patient.  

#This method might still be improved and future work is refrenced in the ReadMe file. 

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy.stats import iqr
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import shap
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


Train_Data = pd.read_csv("C:/Users/NM/Documents/Genes/PPMI_Genes_2.csv", index_col ='ID')

#Group by patient to prevent overfitting with longitudinal patient visits being shared between train, test, validate sets 
Data=Train_Data.T
Data['Number'] = Data.groupby('patient').cumcount().add(1)
Data = Data[Data['QC'] != 0]

#Update filter with degs
Data = Data[[ INSERT DEG LIST, pd, patient]]

groups = Data['patient']
degs = Data.drop(columns= ['patient'])
degs = degs.dropna()

x, y = degs.drop(columns=["pd"]), degs["pd"]

#Set the RF, XGboost, and SVM parameters. 
rf_param_dist = {
    'n_estimators': sp_randint(500, 1500),
    'max_depth': [None] + list(sp_randint(1, 10).rvs(size=10)),
    'min_samples_split': sp_randint(2, 10),
}

xgb_param_dist = {
    'n_estimators': sp_randint(100, 1000),
    'max_depth': sp_randint(1, 10),
    'learning_rate': sp_uniform(0.01, 0.2),
}

svm_param_dist = {
    'C': sp_uniform(0.1, 10),
    'kernel': ['linear', 'rbf'],
}

models_param_dist = [(RandomForestClassifier(), rf_param_dist),
                     (xgb.XGBClassifier(eval_metric='logloss'), xgb_param_dist),
                     (SVC(probability=True), svm_param_dist)]


# Preserve patient groups when splitting
group_kfold = GroupKFold(n_splits=10)


# Use group_kfold.split() instead of train_test_split()
for train_index, test_index in group_kfold.split(x, y, groups):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Feature scaling
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)

    # Upsampling
    print("Before Upsampling:-")
    print(Counter(y_train))

    oversample = SMOTE()
    x_train_ov, y_train_ov = oversample.fit_resample(x_train, y_train)

    print("After Upsampling:-")
    print(Counter(y_train_ov))

    feature_importance = np.zeros(x_train_ov.shape[1])
    outlier_counts = np.zeros(x_train_ov.shape[1])
    roc_auc_scores = []

    # Randomized Search
    for model_class, param_dist in models_param_dist:
        rand_search = RandomizedSearchCV(model_class, param_dist, n_iter=100, cv=5, scoring='roc_auc', n_jobs=-1)
        rand_search.fit(x_train_ov, y_train_ov)
        best_params = rand_search.best_params_
        best_score = rand_search.best_score_
        print(f"Best parameters for {model_class.__class__.__name__}: {best_params}")
        print(f"Best ROC-AUC score: {best_score}\n")

        model_class.set_params(**best_params)
        model = model_class
        model.fit(x_train_ov, y_train_ov)
        Y_pred = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, Y_pred)
        roc_auc_scores.append(roc_auc)

        # Generate SHAP values
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_test, plot_type="bar")


    print("ROC AUC score:", np.median(roc_auc_scores))
