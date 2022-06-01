# src/evaluate_regression.py
import pandas as pd
from datetime import timedelta 
import numpy as np
import config
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import config
import os
import argparse
import model_dispatcher

## PREPROCESSING TEST DATA
def patient_cutoff(df, cutoff_year, cutoff_visits):
    # patients must have this many years of data to be included.
    frames = []
    id_list = df.eye_id.unique()
    for eye in id_list:
        pdf = df[df.eye_id == eye]
        dates = (pd.to_datetime(pdf.admission_date)).to_list()
        if ((dates[-1] - dates[0]).days)/365 >= cutoff_year and len(pdf)>=cutoff_visits: 
            frames.append(pdf)
    return pd.concat(frames)

def cut_time(df, cutoff_time):
        # shortens a patient's dataframe to x years after initiation.
        frames = []
        id_list = df.eye_id.unique()
        for eye in id_list:
            pdf = df[df.eye_id == eye]
            pdf.admission_date = pd.to_datetime(pdf.admission_date)
            dates = pdf['admission_date'].to_list()
            first = pd.to_datetime(dates[0])
            cutoff = first + timedelta(days=cutoff_time*365)
            pdf = pdf[pdf['admission_date'] <= cutoff]
            if len(pdf) > 0: frames.append(pdf)
        return pd.concat(frames)
    
def impute_pdf(df):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index
    imputed_df.fillna(0, inplace=True)
    return imputed_df

def column_names(i):
    return [f'va_{i}', f'int_{i}']

def column_builder(i):
    lst = []
    for visits in range(1, i+1):
        lst.extend(column_names(visits))
    lst.append('mean_vision'), lst.append('std_vision')
    lst.append('target_va')
    lst.remove('int_1')
    return lst

def reshape_pdf(pdf, n_visits):
    nums, columns = [], column_builder(n_visits)
    pdf.fillna(0, inplace=True)
    for i in range(n_visits): 
        nums.append(pdf.visual_acuity.iloc[i])
        if i != 0: nums.append((pdf.admission_date.iloc[i] - pdf.admission_date.iloc[i-1]).days)
    if n_visits > 6: nums.append(np.mean(pdf.visual_acuity))
    else: nums.append(np.mean(pdf.visual_acuity.iloc[:n_visits+1]))
    if n_visits > 3: nums.append(np.std(pdf.visual_acuity))
    else: nums.append(np.std(pdf.visual_acuity.iloc[:n_visits+1]))
    nums.append(pdf.visual_acuity.iloc[-1])
    return pd.DataFrame(data=[nums], columns=columns)

def encode_gender(g):
    return 0 if g == "Male" else 1

def reshape_df(df, n_visits):
    eyes = df.eye_id.unique()
    frames = []
    for eye in eyes:
        pdf = df[df.eye_id == eye]
        try: frames.append(reshape_pdf(pdf, n_visits))
        except: pass
    return pd.concat(frames)

def save_df_patients(n_years, n_visits=4):
    df = pd.read_csv("~/Documents/github/paper/input/raw_test_data_cleaned.csv")
    df.drop(columns=['actual_drug_today', 'next_interval_in_weeks', 'InjNext',
                     'laterality'], inplace=True)
    df["irf"] = 0
    df["srf"] = 0
    df = patient_cutoff(df, n_years, 4)
    df = cut_time(df, n_years)
    df = reshape_df(df, n_visits)
    return df


## TRAINING AND EVALUATION FUNCTIONS
def get_train_test(n_years):
    test_df = save_df_patients(n_years)
    test_df.reset_index(drop=True, inplace=True)

    train_df = pd.read_csv(config.TRAINING_FILE[n_years-1])
    features = ['va_1', 'va_2', 'int_2', 'va_3', 'int_3', 'va_4', 'int_4', 
                'mean_vision', 'std_vision', 'target_va']
    train_df = train_df[features]
    return train_df, test_df

tabnet_params = {"optimizer_fn":torch.optim.Adam,
                 "verbose":0,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax' # "sparsemax"
                }

def score(model, X, y, cv=5, scoring='neg_mean_squared_error'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return np.mean(scores), np.std(scores)

def run(df, clf):
    # create inputs and targets
    X, y = df.drop(columns=['target_va']).values, df.target_va.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    # score the model
    kfold_tabnet(clf, X, y.reshape(-1, 1))

def kfold_tabnet(clf, X, y):
    rmses = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        rmse = fit_tabnet(clf, X_train, y_train, X_test, y_test)
        rmses.append(rmse)
    final_rmse, rmse_std = np.round(np.mean(rmses), 2), np.round(np.std(rmses), 2)
    print(f"Train-set RMSE: mean={final_rmse}, std={rmse_std}")
    
def fit_tabnet(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
            eval_metric=['rmse'], patience=1000, max_epochs=10000)
    preds = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

def evaluate(test_df, clf):
    # create inputs and targets
    X, y = test_df.drop(columns=['target_va']).values, test_df.target_va.values
    # find rmse 
    preds = clf.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"The RMSE on the test-set was {rmse} logMAR letters.")


if __name__ == "__main__":
    # initialise ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments
    parser.add_argument("--years",type=int)
    # read the arguments from the command line
    args = parser.parse_args()
    # run the year specified by the command line arguments
    tabnet_params = {"optimizer_fn":torch.optim.Adam,
                 "verbose":0,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax' # "sparsemax"
                }
    train_df, test_df = get_train_test(args.years)
    clf = TabNetRegressor(**tabnet_params)
    run(train_df, clf)
    evaluate(test_df, clf)