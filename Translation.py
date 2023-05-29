import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import iqr
from sklearn.metrics import roc_auc_score

Train_Data = pd.read_csv("C:/Users/NM/Documents/Genes/PPMI_Genes_2.csv", index_col ='ID')
Data = Train_Data.T

#Drop QC fails
Data = Data[Data['QC'] != 0]

#define target
y = Data["pd"]
Data = Data.drop(columns=["pd"])

#This is the number of features
F = Data.shape[1] 

#Preprocess the data
is_outlier = np.zeros((20, 100, F))
importance = np.zeros((20, 100, F))
thr = np.zeros((20, 100))
percentage_outlier = np.zeros((20, F))
is_selected = np.zeros((20, F, 100))
ROCAUC = np.zeros((20, 10, 100))
m_ROCAUC_r = np.zeros(100)
m_ROCAUC_C = np.zeros(100)
count_selected = np.zeros(F)

#Start the loop, stratifiedkfold, split, RF, select features, XGB, select features, and returned a validated set
for r in range(20):
    skf1 = StratifiedKFold(n_splits=10, shuffle=True, random_state=r)
    for k, (train_index, validation_index) in enumerate(skf1.split(Data, y)):
        training_set_x, validation_set_x = Data.iloc[train_index], Data.iloc[validation_index]
        training_set_y, validation_set_y = y.iloc[train_index], y.iloc[validation_index]
        
        for s in range(100):
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            training_set_index, _ = list(skf2.split(training_set_x, training_set_y))[0]
            new_training_set_x, new_training_set_y = training_set_x.iloc[training_set_index], training_set_y.iloc[training_set_index]
            
            rf = RandomForestRegressor(n_estimators=1000)
            rf.fit(new_training_set_x, new_training_set_y)
            
            for f in range(F):
                importance[r, s, f] = rf.feature_importances_[f]
            
            thr[r, s] = np.median(importance[r, s, :]) + 1.5 * iqr(importance[r, s, :])
            
            for f in range(F):
                is_outlier[r, s, f] = int(importance[r, s, f] > thr[r, s])
        
        for f in range(F):
            percentage_outlier[r, f] = is_outlier[r, :, f].sum()
        
        for C in range(100):
            for f in range(F):
                is_selected[r, f, C] = int(percentage_outlier[r, f] > C)
            
            selected_features = [i for i in range(F) if is_selected[r, i, C] == 1]
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(training_set_x.iloc[:, selected_features], training_set_y)
            
            ROCAUC[r, k, C] = roc_auc_score(validation_set_y, xgb_model.predict(validation_set_x.iloc[:, selected_features]))
  
#View the results
for C in range(100):
    m_ROCAUC_r[C] = np.median(ROCAUC[:, :, C])
    m_ROCAUC_C[C] = np.median(m_ROCAUC_r)

C_star = np.argmax(m_ROCAUC_C)

for f in range(F):
    for r in range(20):
        count_selected[f] += is_selected[r, f, C_star]
