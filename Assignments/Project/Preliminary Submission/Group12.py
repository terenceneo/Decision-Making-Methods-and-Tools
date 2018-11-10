import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics

import decimal
import random

import numpy as np
import pandas as pd

from numpy.random import seed
from tensorflow import set_random_seed

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from datetime import datetime, timedelta

'''
Model summary 
This model aims to trade three currency pairs: EURUSD, GBPUSD, USDCHF hourly
using open, close, high, low data of the pair itself and its correlated pairs.

The model investigates 6 machine learning models (LSTM Neural Network, Logistic Regression, 
Linear Regression, Support Vector Machine Regression, Random Forest Regression and
Gradient Boosting Regression) to predict the change in close price of the current hour.
For each of these 6 models, the code iterates through different configurations 
(size of training set, number of lagged data used for each prediction) and relies on 
time series cross-validation to find the optimal one.

For each currency pair, 6 predictions using 6 machine learning models are generated 
and the final prediction is obtained from an ensemble method that calculates the 
weighted average of the above results.

The idea about reinforcement is also implemented using the retraining approach.
This model is retrained every 12 hours and each time, the lastest 10 percent of
the training set is oversampled to make the model more aware of recent data.
'''

class EnsembleTrading(QCAlgorithm):
    
    # This function is to retrieve historical data and calculate change for each feature.
    # If is_time_delta is true, retrieve data for the last 'count' weeks to prepare for training set.
    # Otherwise, retrieve the last 'count' data points to prepare data for prediction
    def retrieve_data(self, count, is_time_delta=True):
        if is_time_delta:
            df = self.History(self.Securities.Keys, timedelta(weeks=count), Resolution.Hour)
        else:
            df = self.History(self.Securities.Keys, count, Resolution.Hour)
        
        # Calculate the change    
        df = df.loc[:,self.features]
        for feature in self.features:
            df[feature+'_shifted'] = df[feature].groupby('symbol').shift()
            df[feature+'_change'] = df[feature]/df[feature+'_shifted'] - 1
        df.dropna(axis=0, inplace=True)
        df = df.loc[:, [feature+'_change' for feature in self.features]]
        return df
    
    # Transform the original dataframe to include columns for lagged data
    # The output is a dataframe with each row displaying data for one point of time
    # and muti-index in columns (the first level is different pairs and each pair
    # has several columns of different features such as open, close, high, low)
    def windowing(self, df, window):
        windowed_df = df.copy()
        
        # Add data for the last 'window' hours
        for i in range(1, window+1):
            for feature in self.features:
                col = feature + '_change_-' + str(i)
                windowed_df[col] = windowed_df[feature+'_change'].groupby('symbol').shift(i)
        windowed_df = windowed_df.dropna(axis=0)
        windowed_df = windowed_df.T.stack().unstack(0)
        
        # Duplicate the latest 10% of training data for reinforcement purpose
        curr_length = len(windowed_df)
        len_of_duplicate = int(0.1*curr_length)
        duplicate = windowed_df.iloc[-len_of_duplicate:,:]
        windowed_df = pd.concat([windowed_df, duplicate])
        
        return windowed_df
    
    # Prepare training data for each currency pair
    def generate_Xy(self, windowed_df, symbol):
        cols_to_take = [col for col in windowed_df.columns.get_level_values(1) if '-' in col]
        X = windowed_df.loc[:, (slice(None), cols_to_take)]
        y = windowed_df.loc[:, (symbol, 'close_change')]
        return X, y
    
    # Get accuracy score based on Up and Down signals
    def accuracy_metric(self, y_test, y_pred):
        multiplied = y_test*y_pred
        
        # multiplied > 0 if prediction and actual prices are in the same direction
        correct_sign_count = len(multiplied[multiplied > 0])
        total_count = len(y_test)
        return correct_sign_count/total_count
        
    # Create LSTM model for one currency pair
    def create_LSTM(self, X, y, symbol, week, window):
        num_cols = len(X.columns)
        
        # Transform data to fit the required shape
        X = np.array(X)
        y = np.array(y)
        
        O_cells = 100
        O_epochs = 100
        
        X_data1 = np.reshape(X, (X.shape[0],1,X.shape[1]))
        # setting up the neural network step by step
        model = Sequential()
        model.add(LSTM(O_cells, input_shape = (None,num_cols), return_sequences = True))
        model.add(Dropout(0.10))
        model.add(LSTM(O_cells,return_sequences = True))
        model.add(LSTM(O_cells))
        model.add(Dropout(0.10))
        model.add(Dense(1))
        model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
        model.fit(X_data1,y,epochs=O_epochs,verbose=0)
        self.possible_models[symbol][0][(week, window)] = model
    
    def Initialize(self):
        seed(12345) # to make models stable
        set_random_seed(12345)
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetStartDate(2018,10,26)  
        self.SetEndDate(2018,11,2)    
        self.SetCash(1000000)          
        
        self.symbols = ["EURUSD", "GBPUSD", "USDCHF"] # currency pairs used to trade
                        
        self.features = ['close', 'open', 'high', 'low'] # attributes used for prediction
        
        self.long_list = []
        self.short_list = []
        
        self.possible_weeks = [1,2] # Possible numbers of weeks used for training data set
        self.possible_windows = [6, 12] # Possible numbers of lagged hours used for each prediction
        
        # dictionary of multiple forex, each forex with a list of possible models
        self.possible_models = {}
        
        # dictionary of multiple forex
        # each forex with a list of the best configurations for each of the 6 models
        self.best_models = {}
        
        self.tcsv = TimeSeriesSplit(n_splits=2) # to do windowing
        
        self.trained = False # to indicate if the model has already been trained
        self.hours_count = 0 # to count the order of current hours in the 12-hour period, range of (0, 12) inclusive
        
        self.total_count = 0
        
        # dictionary of multiple forex 
        # each with all possible configurations and scores of the 6 models used
        self.grand_total_scoring = {}
        
        # Add in logistic regression model
        self.logistic = LogisticRegression()
        
        # Initialize variables to store model configurations and scores for each forex
        # Add all the forex into the model
        for symbol in self.symbols:
            self.grand_total_scoring[symbol] = []
            self.AddForex(symbol, Resolution.Hour, Market.Oanda)
            self.possible_models[symbol] = [
                {}, # this dictionary is for LSTM, which needs to be created separately
                LinearRegression(),  
                SVR(), 
                RandomForestRegressor(random_state=12345), 
                GradientBoostingRegressor(random_state=12345),
                self.logistic
            ]
            self.best_models[symbol] = {}
        
    def OnData(self, data):
        self.total_count += 1
        self.hours_count += 1 # Update the count for current hour
        
        # Train a new model if no model has been trained or the old model has been used for maximum hours
        if not self.trained or self.hours_count == 12:
            self.hours_count = 0  # Set count for current hour back to 0
            
            max_possible_weeks = max(self.possible_weeks)
            max_possible_windows = max(self.possible_windows)
            grand_df = self.retrieve_data(max_possible_weeks)
            grand_df = self.windowing(grand_df, max_possible_windows)
            len_df_per_forex = len(grand_df)
            
            # Reset variables that store model configurations and scores for each forex
            for symbol in self.symbols:
                self.grand_total_scoring[symbol] = []
                self.best_models[symbol] = {}
            
            # Loop through different numbers of weeks to find the optimal size for training data
            for week in self.possible_weeks:
                slice_row = int(len_df_per_forex/max_possible_weeks*week)
                
                # Loop through different numbers of lagged data for prediction
                for window in self.possible_windows:
                    df = grand_df.tail(slice_row).head(slice_row-window-1)
                    slice_column = [feature+'_change' for feature in self.features]
                    slice_column += [feature+'_change_-'+str(w) for w in range(1,window+1) for feature in self.features]
                    windowed_df = df.loc[:, (slice(None), slice_column)]
                    
                    # Loop through all currency pairs, for each pair apply different models 
                    # with different configurations and get the accuracy
                    for symbol in self.symbols:
                        X, y = self.generate_Xy(windowed_df, symbol)
                        
                        ## Add LSTM to the list of all possible models
                        self.create_LSTM(X, y, symbol, week, window)
                        
                        for model_idx, model in enumerate(self.possible_models[symbol]):
                            scores = [] # to store scores for different time series cross validation
                            
                            # Logistic Regression model
                            if model == self.logistic:
                                y_logs = y.map(lambda x: 1 if x>0 else -1)
                                for train_index, test_index in self.tcsv.split(X):
                                    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                                    y_train, y_test = y_logs.iloc[train_index], y_logs.iloc[test_index]
                                    if (len(X_train) > 0 and len(X_test) > 0):
                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)
                                        accuracy = self.accuracy_metric(y_test, y_pred)
                                        scores.append(accuracy)
                            
                            # LSTM model            
                            elif model_idx == 0: 
                                # reshaping data for LSTM model at index 0
                                X_lstm = np.array(X)
                                y_lstm = np.array(y)
                                for train_index, test_index in self.tcsv.split(X):
                                    X_train, X_test = X_lstm[train_index], X_lstm[test_index]
                                    y_train, y_test = y_lstm[train_index], y_lstm[test_index]
                                    
                                    X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
                                    X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
                                    if (len(X_train) > 0 and len(X_test) > 0):
                                        modelLSTM = model[(week, window)]
                                        modelLSTM.fit(X_train, y_train)
                                        y_pred = modelLSTM.predict(X_test)
                                        accuracy = self.accuracy_metric(y_test, y_pred)
                                        scores.append(accuracy)
                            
                            # Other regression models            
                            else:
                                for train_index, test_index in self.tcsv.split(X):
                                    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                                    if (len(X_train) > 0 and len(X_test) > 0):
                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)
                                        accuracy = self.accuracy_metric(y_test, y_pred)
                                        scores.append(accuracy)
                                        
                            # Take average score for different cross validations
                            model_score = np.mean(scores)
                            
                            # Add model configuration and corresponding score into general list
                            if symbol not in self.grand_total_scoring:
                                self.grand_total_scoring[symbol] = []
                            else:
                                self.grand_total_scoring[symbol].append((week, window, model_idx, model_score))
            
            
            # Loop through all currency pairs, in each pair find the optimal configuration for each 
            # of 6 models and fit the model again with optimal configuration
            for symbol in self.symbols:
                best_model_config = {}
                
                # Find the optimal configuration for each model and store in best_model_config
                for config in self.grand_total_scoring[symbol]:
                    currweek, currwindow, model_index, score = config
                    if model_index not in best_model_config:
                        best_model_config[model_index]=(currweek, currwindow, score)
                    # Update if the score is higher
                    else: 
                        if (score > best_model_config[model_index][2]):
                            best_model_config[model_index]=(currweek, currwindow, score)

                # Fit optimal models again
                for model_index in range(len(self.possible_models[symbol])):
                    optimal_week, optimal_window, optimal_score = best_model_config[model_index]
                    
                    optimal_slice_row = int(len_df_per_forex/max_possible_weeks*optimal_week)
                    optimal_df = grand_df.tail(optimal_slice_row)
                
                    optimal_slice_column = [feature+'_change' for feature in self.features]
                    optimal_slice_column += [feature+'_change_-'+str(w) for w in range(1,optimal_window+1) for feature in self.features]
                    optimal_windowed_df = optimal_df.loc[:, (slice(None), optimal_slice_column)]
                    X, y = self.generate_Xy(optimal_windowed_df, symbol)
                    
                    if model_index != 0:
                        current_model = clone(self.possible_models[symbol][model_index])
                    else:
                        current_model = self.possible_models[symbol][model_index]
                    
                    # Logistic Regression model
                    if str(type(current_model)) == "<class 'sklearn.linear_model.logistic.LogisticRegression'>": 
                        y=y.map(lambda x: 1 if x>0 else -1) # logistic regression requires binary classification
                    
                    # All models but not LSTM
                    if model_index != 0:
                        # Fit the model with proper data
                        current_model.fit(X, y)
                        # Store the model with the number of windowing and its accuracy in the dictionary
                        self.best_models[symbol][model_index] = (current_model, optimal_window, optimal_score) 
                    
                    # LSTM model
                    else:
                        X = np.array(X)
                        y = np.array(y)
                        X = np.reshape(X, (X.shape[0],1,X.shape[1]))
                        current_lstm_model = current_model[(optimal_week, optimal_window)]
                        current_lstm_model.fit(X, y)
                        self.best_models[symbol][model_index] = (current_lstm_model, optimal_window, optimal_score)
                    
            self.trained = True
        
        # Prepare data for prediction
        maximum_window = 0
        for symbol in self.best_models:
            for model_index in self.best_models[symbol]:
                if self.best_models[symbol][model_index][1] > maximum_window:
                    maximum_window = self.best_models[symbol][model_index][1]
        
        new_data = self.retrieve_data(maximum_window+4, is_time_delta=False) 
        weighted_prediction = 0 # Initially, set prediction to be 0
        
        # For each forex, get the weighted prediction based on the accuracy on the validation set
        for symbol in self.symbols:
            for model, window, accuracy in self.best_models[symbol].values():
                X = self.windowing(new_data, window-1).iloc[-1,:].values.reshape(1, -1)
                
                if str(type(model)) == "<class 'keras.engine.sequential.Sequential'>":
                    X = np.reshape(X, (X.shape[0],1,X.shape[1]))
                
                prediction = model.predict(X)[-1]
                if prediction > 0: # the price is predicted to increase
                    weighted_prediction += 1.0*accuracy
                else: # the price is predicted to decrease
                    weighted_prediction += (-1.0)*accuracy
                    
            weight = 0.9/len(self.symbols) # the holding to be set for each currency pair
            prediction = weighted_prediction
            
            # Get the current price of the current forex
            price = data[symbol].Close
            
            # Entry /Exit Conditions for trading
            
            # Buy the currency
            if prediction > 0.00005  and symbol not in self.long_list and symbol not in self.short_list:
                self.SetHoldings(symbol, weight)
                self.long_list.append(symbol)
            
            # Stop loss - Take profit
            if symbol in self.long_list:
                cost_basis = self.Portfolio[symbol].AveragePrice
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                    # if true then sell
                    self.SetHoldings(symbol, 0)
                    self.long_list.remove(symbol)
                    
            # Short selling
            if prediction < -0.00005 and symbol not in self.long_list and symbol not in self.short_list:
                self.SetHoldings(symbol, -weight)
                self.short_list.append(symbol)
            
            # Stop loss - Take profit
            if symbol in self.short_list:
                cost_basis = self.Portfolio[symbol].AveragePrice
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                    # if true then buy back
                    self.SetHoldings(symbol, 0)
                    self.short_list.remove(symbol)
            
        # Liquidate on the last trading day
        if str(self.Time.date()) == '2018-11-3':
            self.Liquidate()