from sklearn import ensemble
from sklearn import linear_model
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

tabnet_params = {"optimizer_fn":torch.optim.Adam,
                 "verbose":0,
                 "optimizer_params":dict(lr=2e-2),
                 "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                 "gamma":0.9},
                 "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                 "mask_type":'entmax' # "sparsemax"
                }


classify_models = {
 "rf": ensemble.RandomForestClassifier(max_depth=40, n_estimators=250, 
                                       max_features=10, random_state=42),
 "gb": ensemble.GradientBoostingClassifier(),
 "lr": linear_model.LogisticRegression(),
 "tn": TabNetClassifier(**tabnet_params)}

regression_models = {
 "rf": ensemble.RandomForestRegressor(),
 "gb": ensemble.GradientBoostingRegressor(),
 "lr": linear_model.LinearRegression(),
 "tn": TabNetRegressor(**tabnet_params)}