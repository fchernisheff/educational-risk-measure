#Loading packages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import joblib
import os

#Downloading dataset

raw_df = pd.read_csv('hybrid_student_performance_1200.csv')
raw_df = raw_df.drop(columns=["student_id", "timestamp"])
raw_df.fillna("Unknown", inplace=True)

#Classifying columns

TARGET = "performance_risk_level"

X = raw_df.drop(columns=[TARGET])
y = raw_df[TARGET]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
categorical_cols = X.select_dtypes(include=["object"]).columns

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

#Splitting to test and train sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

#XGBClassifier hyperparameter tuning

def objective_xgb(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "enable_categorical": False
    }
    XGBmodel = XGBClassifier(**params)
    XGBmodel.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = XGBmodel.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy
    
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=200)

#LGBMClassifier hyperparameter tuning

def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "objective": "multiclass",
        "num_class": 3,
        "random_state": 42
    }
    LGBMmodel = LGBMClassifier(**params, verbosity=-1)
    LGBMmodel.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)]
    )
    preds = LGBMmodel.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=200)

#Catboost hyperparameter tuning

def objective_catboost(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 200, 1200),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "verbose": 0,
        "random_seed": 42
    }
    CBmodel = CatBoostClassifier(**params)
    CBmodel.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        verbose=False
    )
    preds = CBmodel.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_catboost, n_trials=50)

#Ensemble model creating

best_xgb = XGBClassifier(
    **study_xgb.best_params,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

best_lgbm = LGBMClassifier(
    **study_lgbm.best_params,
    objective="multiclass",
    num_class=3,
    random_state=42
)
best_cat = CatBoostClassifier(
    **study_cat.best_params,
    loss_function="MultiClass",
    verbose=0,
    random_seed=42
)
ensemble = VotingClassifier(
    estimators=[
        ("xgb", best_xgb),
        ("lgbm", best_lgbm),
        ("cat", best_cat)
    ],
    voting="soft"
)
ensemble.fit(X_train, y_train)

#Saving model

os.makedirs("model", exist_ok=True)

joblib.dump(ensemble, "model/ensemble_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(label_encoder, "model/target_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/features.pkl")

