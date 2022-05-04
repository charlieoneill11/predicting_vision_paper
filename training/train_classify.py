# src/train_classify.py
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

def score(model, X, y, cv=5, scoring='accuracy'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return np.mean(scores), np.std(scores)

def label_ovc(row): return 1 if (row.target_va - row.va_1) >= 0 else 0

def label_ovc_multi(row):
    final_bcva = row.target_va
    toRtn = 0
    if final_bcva < 60: 
        if final_bcva < 40: toRtn = 0
        else: toRtn = 1
    else:
        if final_bcva < 80: toRtn = 2
        else: toRtn = 3
    return toRtn

def run(model, years_visits, multi=False, visits=True):
    # read the training data with folds
    if visits: 
        df = pd.read_csv(config.MULTIVISITS_FILE[years_visits-2])
        print(f"NUMBER OF INPUT VISITS: {years_visits}")
    else: df = pd.read_csv(config.TRAINING_FILE[years_visits-1])
    # create the target variable
    if multi: df['outcome'] = df.apply(lambda row: label_ovc_multi(row), axis=1)
    else: df['outcome'] = df.apply(lambda row: label_ovc(row), axis=1)
    # create inputs and targets
    X, y = df.drop(columns=['target_va', 'outcome']).values, df.outcome.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    # call the model
    clf = model_dispatcher.classify_models[model]
    # score the model (default is accuracy)
    if model != "tn": 
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)
        if multi: auc_mean, auc_std = score(clf, X, y, scoring='roc_auc_ovr_weighted')
        else: auc_mean, auc_std = score(clf, X, y, scoring='roc_auc')
        acc_mean, acc_std = score(clf, X, y, scoring='accuracy')
        print(f"AUC: mean={np.round(auc_mean, 2)}, std={np.round(auc_std, 2)}")
        print(f"Accuracy: mean={np.round(100*acc_mean, 2)}%, std={np.round(100*acc_std, 2)}")
    else: kfold_tabnet(clf, X, y, multi)
    #joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{model}.bin"))

def kfold_tabnet(clf, X, y, multi):
    aucs, accuracies = [], []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        auc, accuracy = fit_tabnet(clf, X_train, y_train, X_test, y_test, multi)
        aucs.append(auc), accuracies.append(accuracy)
    auc, auc_std = np.round(np.mean(aucs), 4), np.round(np.std(aucs), 4)
    accuracy, acc_std = np.round(np.mean(accuracies), 2), np.round(np.std(accuracies), 2)
    print(f"AUC: mean={auc}, std={auc_std}")
    print(f"Accuracy: mean={accuracy}%, std={acc_std}")

def fit_tabnet(clf, X_train, y_train, X_test, y_test, multi):
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
            eval_metric=['accuracy'], patience=1000, max_epochs=10000)
    if multi: preds_proba = clf.predict_proba(X_test)
    else: preds_proba = clf.predict_proba(X_test)[:,1]
    test_auc = roc_auc_score(y_score=preds_proba, y_true=y_test,
                             average="weighted", multi_class="ovr")
    preds = np.round(clf.predict(X_test), 4)
    accuracy = np.round(100*accuracy_score(y_test, preds), 2)
    return test_auc, accuracy

if __name__ == "__main__":
    # initialise ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments
    parser.add_argument("--model",type=str)
    parser.add_argument("--years_visits",type=int)
    parser.add_argument("--multi",type=bool)
    parser.add_argument("--visits",type=bool)
    # read the arguments from the command line
    args = parser.parse_args()
    # run the fold specified by the command line arguments
    run(model=args.model, years_visits=args.years_visits, multi=args.multi, 
        visits=args.visits)