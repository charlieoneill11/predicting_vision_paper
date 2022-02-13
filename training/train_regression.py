# src/train_regression.py
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config
import os
import argparse
import model_dispatcher

def score(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return np.mean(scores), np.std(scores)

def run(model, years):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE[years-1])
    # create inputs and targets
    X, y = df.drop(columns=['target_va']).values, df.target_va.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    # call the model
    clf = model_dispatcher.regression_models[model]
    # score the model (default is accuracy)
    if model != "tn": 
        clf.fit(X_train, y_train)
        mean, std = score(clf, X, y)
        mean, std = np.sqrt(abs(mean)), np.sqrt(std)
        print(f"RMSE: mean={np.round(mean, 2)}, std={np.round(std, 2)}")
    else: kfold_tabnet(clf, X, y.reshape(-1, 1))
    #joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{model}.bin"))

def kfold_tabnet(clf, X, y):
    rmses = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        rmse = fit_tabnet(clf, X_train, y_train, X_test, y_test)
        rmses.append(rmse)
    final_rmse, rmse_std = np.round(np.mean(rmses), 2), np.round(np.std(rmses), 2)
    print(f"RMSE: mean={final_rmse}, std={rmse_std}")

def fit_tabnet(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
            eval_metric=['rmse'], patience=1000, max_epochs=10000)
    preds = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

if __name__ == "__main__":
    # initialise ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments
    parser.add_argument("--model",type=str)
    parser.add_argument("--years",type=int)
    # read the arguments from the command line
    args = parser.parse_args()
    # run the fold specified by the command line arguments
    run(model=args.model, years=args.years)