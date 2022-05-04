# src/evaluate_classify.py
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import config
import os
import argparse
import model_dispatcher

def label_ovc(row): return 1 if (row.target_va - row.first_va) >= 0 else 0

def evaluate(model, years):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE[years-1])
    # create the target variable
    df['outcome'] = df.apply(lambda row: label_ovc(row), axis=1)
    # create inputs and targets
    X, y = df.drop(columns=['target_va', 'outcome']).values, df.outcome.values
    # call the model
    clf = model_dispatcher.classify_models[model]
    # score the model (default is accuracy)
    if model != "tn": clf.fit(X, y)
    else: clf.fit(X, y, eval_set=[(X, y)],
                  eval_metric=['accuracy'], patience=1000, max_epochs=10000)
    # get the test data
    df_test = pd.read_csv(config.EVALUATION_FILE[years-1])
    df_test['outcome'] = df_test.apply(lambda row: label_ovc(row), axis=1)
    X, y = df_test.drop(columns=['target_va', 'outcome']).values, df_test.outcome.values
    # test auc and accuracy
    auc_preds = clf.predict_proba(X)
    test_auc = np.round(roc_auc_score(y_score=auc_preds[:,1], y_true=y), 2)
    accuracy_preds = clf.predict(X)
    test_accuracy = np.round(100*accuracy_score(y, accuracy_preds), 2)
    print(f"TEST: AUC = {test_auc}, Accuracy = {test_accuracy}%")

if __name__ == "__main__":
    # initialise ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments
    parser.add_argument("--model",type=str)
    parser.add_argument("--years",type=int)
    # read the arguments from the command line
    args = parser.parse_args()
    # run the fold specified by the command line arguments
    evaluate(model=args.model, years=args.years)

# TO-DO: Shadings for asggregated temporal plot