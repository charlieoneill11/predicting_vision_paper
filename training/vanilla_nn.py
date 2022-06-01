import warnings
warnings.filterwarnings('ignore')
import config
import os
import argparse
import model_dispatcher
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import datetime
from torch.autograd import Variable

def train_test_splitter(train_df):
    X, y = train_df.drop(columns=['target_va']).values, train_df.target_va.values
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test 

def create_dataset(X_train, y_train, X_test, y_test):
    train_data, test_data = [], []
    for i in range(len(X_train)):
        train_data.append([X_train[i], 
                        y_train[i]])
    for i in range(len(X_test)):
        test_data.append([X_test[i], 
                        y_test[i]])
    return train_data, test_data 

class SimpleNet(nn.Module):
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(17, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.25)
        self.double()
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = x.view(x.size(0), -1)      
        out = self.out(x)
        return out

def training_loop(model, optimiser, loss_fn, n_epochs, train_loader):
    model.train()
    for n in range(1, n_epochs+1):
        loss_train = 0.0
        for i, (eyes, labels) in enumerate(train_loader):
            b_x = Variable(eyes)
            b_y = Variable(labels)
            output = model(b_x)
            loss = loss_fn(output, b_y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_train += loss.item()
            preds = model(eyes).detach().numpy()
            rmse_train = np.sqrt(mean_squared_error(preds, labels))
            
        # validation metrics
        model.eval()
        with torch.no_grad():
            for eyes, labels in test_loader:
                preds = model(eyes)
                rmse = np.sqrt(mean_squared_error(preds, labels))
            
        if n == 1 or n % 1000 == 0:
            print(f'Epoch [{n}/{n_epochs}], Train Loss: {np.round(loss.item(), 2)}, Train RMSE: {np.round(rmse_train, 2)}, Validation RMSE = {rmse}')

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    # initialise ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    # add the different arguments
    parser.add_argument("--years",type=int)
    # read the arguments from the command line
    args = parser.parse_args()
    # import dataframe
    train_df = pd.read_csv(config.TRAINING_FILE[args.years-1])
    # get numpy data
    X_train, X_test, y_train, y_test = train_test_splitter(train_df)
    # create datasets
    train_data, test_data = create_dataset(X_train, y_train, X_test, y_test)
    # create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, 
                                               batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False,
                                              batch_size=100)
    # instantiate mode
    model = SimpleNet()
    model.apply(init_weights) # xavier initialisation
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    # training run
    training_loop(model, optimiser, loss_fn, n_epochs=10000, train_loader=train_loader)
