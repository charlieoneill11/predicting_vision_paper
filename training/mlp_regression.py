import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=4, padding=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=2, kernel_size=4, padding=2)
        self.layer1 = nn.Sequential(self.conv1, nn.ReLU())
        self.layer2 = nn.Sequential(self.conv2, nn.ReLU())         
        # fully connected layer, output 2 classes
        self.layer3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU(),
                                    nn.Linear(64, 16), nn.ReLU(),
                                    nn.Linear(16, 8), nn.ReLU())
        self.out = nn.Linear(8, 1)
    
    def forward(self, x):
        x = self.layer1(x.unsqueeze(1))
        x = self.layer2(x)
        # flatten the output of conv2 to (batch_size, 32)
        x = x.view(x.size(0), -1) 
        x = self.layer3(x)
        output = self.out(x)
        return output

class PytorchKfolds:
    
    def __init__(self, n_epochs=30):
        self.df = pd.read_csv("~/Documents/Github/paper/input/df_2_years.csv")
        self.kdf = self.create_folds(self.df)
        self.n_epochs = n_epochs
        self.loss_fn = nn.MSELoss()
    
    def create_folds(self, df):
        # we create a new column called kfold and fill it with -1
        df["kfold"] = -1
        # the next step is to randomize the rows of the data
        df = df.sample(frac=1).reset_index(drop=True)
        # fetch labels
        y = df.target_va.values
        # initiate the kfold class from model_selection module
        kf = model_selection.KFold(n_splits=5)
        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'kfold'] = f
        return df
    
    def inputs_targets(self, df, fold):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_train.fillna(df_train.mean(), inplace=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        df_valid.fillna(df_valid.mean(), inplace=True)
        X_train = df_train.drop(columns=["target_va", "kfold"]).values
        y_train = df_train.target_va.values
        X_valid = df_valid.drop(columns=["target_va", "kfold"]).values
        y_valid = df_valid.target_va.values
        return X_train, X_valid, y_train, y_valid
    
    def train_test_kfold(self, df, fold):
        X_train, X_test, y_train, y_test = self.inputs_targets(df, fold)
        # scale the data
        ss = StandardScaler()
        mm = MinMaxScaler()
        X_train, X_test = ss.fit_transform(X_train), ss.fit_transform(X_test)
        y_train = mm.fit_transform(y_train.reshape(-1, 1))
        y_test = mm.fit_transform(y_test.reshape(-1, 1))
        # convert to tensors
        X_train_tensors = Variable(torch.Tensor(X_train))
        X_test_tensors = Variable(torch.Tensor(X_test))
        y_train_tensors = Variable(torch.Tensor(y_train))
        y_test_tensors = Variable(torch.Tensor(y_test))
        return X_train_tensors, X_test_tensors, y_train_tensors, y_test_tensors
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test):
        train_data, test_data = [], []
        for i in range(len(X_train)):
            train_data.append([X_train[i].to(torch.float32), 
                               y_train[i].type(torch.float32)])
        for i in range(len(X_test)):
            test_data.append([X_test[i].to(torch.float32), 
                               y_test[i].type(torch.float32)])
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, 
                                               batch_size=64, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, 
                                                  batch_size=len(X_test))
        return train_loader, test_loader
    
    def reset_weights(self, m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
    def rmse_score(self, val_loader, model):
        model.eval()
        rmses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs).detach().numpy()
                labels = labels.detach().numpy()
                rmse = np.sqrt(mean_squared_error(outputs, labels))
                rmses.append(rmse)
            return np.round(np.mean(rmses), 4)
                
    
    def training_loop(self, train_loader, val_loader, verbose=0):
        n_epochs=self.n_epochs
        model=Net()
        model.apply(self.reset_weights)
        optimiser=torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn=self.loss_fn
        model.train()
        count, best_rmse = 0, 1000
        for epoch in range(1, n_epochs + 1):
            loss_train, loss_test = 0.0, 0.0
            for imgs, labels in train_loader:
                b_x = Variable(imgs)   # batch x
                b_y = Variable(labels)   # batch y
                outputs = model(imgs)
                loss = loss_fn(outputs, b_y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                loss_train += loss.item() # .item() is used to escape gradient
            rmse = 100*self.rmse_score(val_loader, model)
            # early stopping
            if (best_rmse < rmse) and count > 40: 
                print("EARLY STOPPING INITIATED.")
                print(f"Best RMSE = {best_rmse}. Epoch = {epoch}.")
                break
            else: 
                if best_rmse > rmse: best_rmse = rmse
                count += 1
            if (epoch == 1 or epoch % (n_epochs/10) == 0) and verbose>1:
                print("Epoch {}, Training Loss {}, RMSE {}%".format(
                    epoch,
                    np.round(loss_train / len(train_loader), 4),
                    np.round(rmse, 2)))
        test_rmse = 100*self.rmse_score(val_loader, model)
        if verbose==1: print(f"Test RMSE = {test_rmse}")
        return test_rmse
                
    def kfold_train(self, verbose=0):
        rmses = []
        for i in range(5):
            if verbose > 0: print(f'FOLD {i}')
            X_train, X_test, y_train, y_test = self.train_test_kfold(self.kdf, i)
            train_loader, val_loader = self.create_dataloaders(X_train, X_test, y_train, y_test)
            test_rmse = self.training_loop(train_loader=train_loader,
                                           val_loader=val_loader, verbose=verbose)
            rmses.append(np.round(test_rmse, 2))
            if verbose > 0: print('--------------------------------')
        if verbose!=-1:
            print("FINAL RESULTS")
            print(f"Mean validation RMSE: {round(np.mean(rmses), 2)} (+/- {round(np.std(rmses), 2)})")
        if verbose==-1: return np.mean(rmses)

if __name__ == "__main__":
    ete = PytorchKfolds(n_epochs=2000)
    ete.kfold_train(verbose=1)