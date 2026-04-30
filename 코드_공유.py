# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.


df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target



X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

####### A 작업자 작업 수행 #######

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 3, 4]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=5
)

dt_grid.fit(X_train, y_train)

dt_best_model = dt_grid.best_estimator_
dt_pred = dt_best_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("DT Best Params:", dt_grid.best_params_)
print("DT Accuracy:", dt_accuracy)



####### B 작업자 작업 수행 #######


####### B 작업자 작업 수행 #######

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [50, 100]
}

xgb_grid = GridSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric='mlogloss'
    ),
    param_grid=xgb_param_grid,
    cv=5
)

xgb_grid.fit(X_train, y_train)

xgb_best_model = xgb_grid.best_estimator_
xgb_pred = xgb_best_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("XGB Best Params:", xgb_grid.best_params_)
print("XGB Accuracy:", xgb_accuracy)