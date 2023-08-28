#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import pickle
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


if not os.path.exists(r'../models'):
    os.makedirs(r'../models')
if not os.path.exists(r'../plots'):
    os.makedirs(r'../plots')


class DLModel:
    """
        Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [235] * 10
        self.L = 0.035
    
    def get_predictions(self, x, Z0=None, w=10, L=None) -> np.ndarray:
    
        Z = Z0 * (1 - np.exp((-L * x) / Z0))
        return Z  

    def calculate_loss(self, Params, X, Y, w=10) -> float:
        
        Z0 = Params[1]
        L = Params[0]
        i = 0
        mse = 0
        for x in X:
            pred = self.get_predictions(x, Z0, w, L)
            mse += (pred - Y[i])**2
            i += 1
        mse = mse/i
        return mse
    
    def save(self, path):
       
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path) -> Union[pd.DataFrame, np.ndarray]:
    
    data = pd.read_csv(r'../data/04_cricket_1999to2011.csv')
    return data


def preprocess_data(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    
    # Drop rows where the value in 'Error.In.Data' is 1
    data = data.loc[data['Error.In.Data'] != 1]
    
    # Drop rows where the value in 'Innings' is 2
    data = data.loc[data['Innings'] != 2]
    
    #selecting specific columns from data which we need
    df = data.iloc[:, [7, 11]]
    
    #subtracting from overs column to get overs reamining
    new = 50 - data.iloc[:,3] 
    
    #converting 'new' to a dataframe and concatenating to main data
    a = pd.DataFrame(new) 
    data_new = pd.concat([a, df], axis=1)
    
    # changing date from dd/mm/yyyy to dd-mm-yyyy format
    data['Date'] = data['Date'].str.replace('/', '-', regex=False)
    
    return data_new


def train_model(data_new: Union[pd.DataFrame, np.ndarray], model: DLModel) -> DLModel:
    
    Z0 = [235]*10
    
    for i in range(1,11):
        wickets = data_new[data_new['Wickets.in.Hand']==i]
        Y = wickets['Runs.Remaining'].values
        x = wickets['Over'].values

        # Initial parameter guess for optimization
        initial_params = [model.L, 235]

        # Define the optimization function (loss function)
        def optimization_function(params):
            return model.calculate_loss(params, x, Y, w=10)

        # Performing optimization using SciPy's minimize function
        result = minimize(optimization_function, initial_params, method='BFGS')

        # Updating the model's parameters with optimized values
        model.L, giru = result.x
        Z0[i-1] = giru
         
    model = [Z0,model.L]
    return model


def plot(model: DLModel, plot_path: str) -> None:
    
    overs_remaining = np.linspace(0, 50, 100) 
    
    # Calculating predictions using the trained model
    Z0 = model[0]
    L = model[1]
    
    plt.figure(figsize=(15, 12))
    
    for i in range(1, 11):
        predictions = Z0[i-1] * (1 - np.exp(-(L * overs_remaining / Z0[i-1])))
        plt.plot(overs_remaining, predictions, label=f'wickets = {i}')
    
    plt.xlabel('Overs Remaining', fontsize=16)
    plt.ylabel('Predicted Runs', fontsize=16)
    plt.title('Run Production Function', fontsize=20, fontweight='bold')
    plt.legend()
    
    plt.savefig(plot_path)
    # Show the plot
    plt.show()


def print_model_params(model: DLModel) -> List[float]:
    
    print('_____________________________________________________________________________________')
    parameters = model
    i = 1
    for param in parameters[0]:
        print(f"Model parameter Z{i}: {param:.2f}")
        i += 1
    print('_____________________________________________________________________________________')
    print(f"Model parameter L: {parameters[1]:.2f}")
    return parameters


def calculate_loss(model: DLModel, data_new: Union[pd.DataFrame, np.ndarray]) -> float:
    
    Z0 = model[0]
    L = model[1]
    mse = 0
    print('_____________________________________________________________________________________')
    for i in range(1,11):
        wickets = data_new[data_new['Wickets.in.Hand']==i]
        Y = wickets['Runs.Remaining'].values
        X = wickets['Over'].values
        pred = Z0[i-1] * (1 - np.exp((-L * X) / Z0[i-1]))
        mse += np.sum((pred - Y)**2)
    mse = mse/len(data_new.values)
    print(f"Mean Squared Error is: {mse:.2f}")
    return mse
    

def main(args):
    """Main Function"""

    data = get_data(args['data_path'])  # Loading the data
    print("Data loaded.")
    
    print('_____________________________________________________________________________________')
    
    # Preprocess the data
    data_new = preprocess_data(data)
    print("Data preprocessed.")
    
    print('_____________________________________________________________________________________')
    
    model1 = DLModel()  # Initializing the model
    model = train_model(data_new, model1)  # Training the model
    model1.save(args['model_path'])  # Saving the model
    
    plot(model,args['plot_path'])  # Plotting the model
    
    # Printing the model parameters
    print_model_params(model)

    # Calculate the normalised squared error
    calculate_loss(model, data_new)


if __name__ == '__main__':
    args = {
        "data_path": r'../data/04_cricket_1999to2011.csv',
        "model_path": r'../models/model.pkl',  # ensure that the path exists
        "plot_path": r'../plots/plot.png',  # ensure that the path exists
    }
    main(args)


# In[ ]:




