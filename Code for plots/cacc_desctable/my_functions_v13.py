# Data fixes before feature engineering --> X and Y are already initialized
# V12: Same as V9, but additional functionality in make_predictions for variable importance analysis
# V13: Same as V12, but number of factors included in model directory, added intercept to X, fixed the covariates sorted portfolios now (used later for interpretation of latent factors and performance benchmark)
import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy.stats import rankdata
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
import keras_tuner as kt
from kerastuner.tuners import Hyperband
import tqdm
from kerastuner.tuners import BayesianOptimization
import random
from scipy.stats.mstats import winsorize
import pdb
import pickle

from sklearn.preprocessing import StandardScaler
import os


# Function to read .csv files. Uses only those that have a price column. Returns list of dataframes, their names
# and the sorted times
def read_files(data_path, remove_coins=[]):
    filenames = os.listdir(data_path)
    filenames.sort()
    # Select only .csv files
    filenames = [file for file in filenames if file[-4:] == ".csv" and file[:-4] not in remove_coins]
    # Read files
    dflist = [pd.read_csv(data_path + "/" + file, index_col="day") for file in filenames]
    for df in dflist: df.index = pd.to_datetime(df.index)
    # Extract all time points in a chronologically sorted manner
    sorted_times = sort_all_times(dflist)
    return dflist, filenames, sorted_times


# Function that takes a list of data frames and returns a chronologically ordered array of all times
# Times need to be in index
def sort_all_times(dflist):
    # Extract list of times from each dataframe
    list_times = [x.index for x in dflist]
    # List of sets of times from each dataframe
    list_times_sets = [set(x) for x in list_times]
    # Unite all sets
    all_times = list_times_sets[0]
    for i in range(len(list_times_sets)):
        all_times = all_times.union(list_times_sets[i])
    # Save as array
    all_times_array = pd.array(all_times)
    # Sort chronologically
    all_times_sorted = pd.array(sorted(all_times_array))

    return all_times_sorted


# Function that takes a list of pandas Series and concatenates them all into one dataframe, where each column is one
# of the original series. Second argument is the list of names of the columns
def combine_returns(list_return_dfs, list_coin_names):
    for i in range(len(list_return_dfs)):
        if i == 0:
            prices_df = list_return_dfs[i]
        else:
            prices_df = pd.concat([prices_df, list_return_dfs[i]], axis=1)
    prices_df.columns = [coin[:-4] for coin in list_coin_names]
    return prices_df


# Function that takes as input a list of dataframes and gives as output a list of all column/covariate names
def all_columns(dflist):
    columns_list = [column for df in dflist for column in df.columns]
    columns_set = set(columns_list)
    columns_final_list = list(columns_set)
    return columns_final_list


# Function that
# takes as input:
# list of dataframes, date as string, minimum number of coins, minimum percentage of observations per
# variable per coin, list of dataframe names
# gives as output:
# List of coin names, List of covariate names, X (returns + covariates), y (returns)
# Here covariates are rank standardized crosssectionally
def prepare_X_y_rank_standardized(dflist, startdate, enddate, filenames, variables):
    sorted_times = sort_all_times(dflist)

    # Cut away all time points before the variable date
    selected_times = sorted_times[sorted_times >= startdate]

    # Cut away all time points after the variable enddate
    selected_times = selected_times[selected_times <= enddate]

    dflist_cut = [df.reindex(selected_times) for df in dflist]

    # If a data point needs to be excluded, we delete all info

    for df in dflist_cut:
        df.loc[df["exclude_training_and_testing"] == "True", ["daily excess return", "close"]] = np.nan
        df.loc[df["exclude_training_and_testing"] == "True", variables] = np.nan

    # Split up excess returns, rest of data and exclusion mask
    list_of_excess_returns = [df["daily excess return"] for df in dflist_cut]
    df_returns = combine_returns(list_of_excess_returns, filenames)
    list_of_close_prices = [df["close"] for df in dflist_cut]
    df_close_prices = combine_returns(list_of_close_prices, filenames)
    list_of_exclusions = [df["exclude_testing"] for df in dflist_cut]
    exclusion_mask = combine_returns(list_of_exclusions, filenames)
    exclusion_mask.replace(np.nan, True, inplace=True)

    list_of_covariates = [df.loc[:, variables] for df in dflist_cut]

    # Working with a list of dataframes is computationally cumbersome. We change to a 3d numpy-array
    data_covariates = np.array(list_of_covariates)

    # Replace NAs of covariates with cross-sectional median of other coins
    data_covariates_nan_median = np.where(np.isnan(data_covariates),
                                          ma.median(ma.array(data_covariates, mask=np.isnan(data_covariates)), axis=0),
                                          data_covariates)

    # Rank normalize covariates to (-1, 1)
    covariates_rank_normalized = 2 * (rankdata(data_covariates_nan_median, axis=0) - 1) \
                                 / (data_covariates.shape[0] - 1) - 1

    # Go back to list of dataframes
    list_covariates_dfs = [pd.DataFrame(data=covariates_rank_normalized[i, :, :],
                                        columns=variables,
                                        index=selected_times)
                           for i in np.arange(covariates_rank_normalized.shape[0])]
    covariates_unstandardized = list_of_covariates
    return df_returns, list_covariates_dfs, covariates_unstandardized, df_close_prices, exclusion_mask


def create_covariates_nn(num_coins, num_covariates, encoding_dim, list_no_hidden, lambda_reg):
    dict_nn = {}
    hidden = [layers.Dense(no_hidden, activation='relu', kernel_regularizer=l2(lambda_reg)) for no_hidden in
              list_no_hidden]
    for i in range(num_coins):
        subdict = {}
        subdict["input"] = keras.Input(shape=(num_covariates,))
        subdict["hidden"] = hidden[0](subdict["input"])
        if len(hidden) > 1:
            for layer in hidden[1:]:
                subdict["hidden"] = layer(subdict["hidden"])
        subdict["output"] = layers.Dense(encoding_dim,
                                         name="betas" + str(i),
                                         kernel_regularizer=l2(lambda_reg))(subdict["hidden"])
        # subdict["model"] = keras.Model(inputs = subdict["input"], outputs = subdict["output"])
        dict_nn["{}".format(i)] = subdict
    return dict_nn


# Create the encoder part for the returns

def create_returns_encoder(shape_input, encoding_dim, lambda_reg):
    dict_encoder = {}
    # These are our inputs for the factors
    dict_encoder["input"] = keras.Input(shape=(shape_input,))

    # "encoded" is the encoded representation of the input
    dict_encoder["output"] = layers.Dense(encoding_dim,
                                          name="Factors",
                                          # activation = "relu",
                                          kernel_regularizer=l2(lambda_reg))(dict_encoder["input"])
    return dict_encoder


# Function that links the parallel networks of the covariate part with the output of the returns part

def link_both_parts(num_coins, input_dimension, num_covariates, encoding_dim, list_no_hidden, lambda_reg=0.01):
    dict_covars = create_covariates_nn(num_coins, num_covariates, encoding_dim, list_no_hidden, lambda_reg)
    dict_dot_product = {}
    encoder = create_returns_encoder(input_dimension, encoding_dim, lambda_reg)
    for i in range(num_coins):
        dict_dot_product[str(i)] = tf.keras.layers.Dot(axes=1)([dict_covars[str(i)]["output"], encoder["output"]])

    concatted = tf.keras.layers.Concatenate()([dict_dot_product[str(i)] for i in range(num_coins)])

    input_layers = [dict_covars[str(i)]["input"] for i in range(num_coins)]
    input_layers.append(encoder["input"])
    full_model = tf.keras.Model(inputs=input_layers,
                                outputs=concatted,
                                name="full_model_{0}_parallel_networks".format(num_coins))
    return full_model


class regression_model:

    def __init__(self, data_path="../Aggregate Data v3", remove_coins=[]):
        self.data_path = data_path
        self.dflist, self.filenames, self.sorted_times = read_files(self.data_path, remove_coins=remove_coins)
        self.variables = ["new_addresses", "active_addresses", "bm", "volumeto", "size", "illiq", "capm beta",
                          "capm alpha", "ivol", "turnover", "rvol", "bid-ask",
                          "detrended turnover", "standard deviation turnover", "rel to high",
                          "volume shock", "r2_1", "r7_2", "r13_2", "r22_2", "r31_2",
                          "r30_14", "r180_60", "var_5"]
        # Even though our data starts on 2017-01-01, we delete the first 180 days because we do not have data
        # there for a lot of the features we engineered ourselves
        self.startdate = pd.to_datetime("2017-01-01") + pd.to_timedelta(180, "day")
        # Last day where we have data for bitcoin
        self.enddate = pd.to_datetime("2022-03-01")
        self.test_percent = 0.5
        self.validation_percent = 1 / 3

    def replace_infty(self):
        dflist = self.dflist
        replaced = []
        for df in dflist:
            new = df.replace(to_replace=[np.inf, -np.inf], value=np.nan)
            replaced.append(new)

        self.dflist = replaced

    def prepare_rank_standardize_data(self):
        self.returns, self.variables_df_list, self.variables_df_list_unstandardized, self.close_prices, self.exclusion_mask = \
            prepare_X_y_rank_standardized(self.dflist, self.startdate, self.enddate, self.filenames, self.variables)


class cond_auto_model():
    def __init__(
            self, model_name, data_path="../Aggregate Data v3", list_covar_hidden_neurons=[32],
            list_factors_hidden_neurons=[5],
            factor_dim=5, predictiondays=30, remove_coins=[], full_input=True, parallel_runs=10):

        self.data_path = data_path
        self.dflist, self.filenames, self.sorted_times = read_files(self.data_path, remove_coins=remove_coins)
        self.variables = ["new_addresses", "active_addresses", "bm", "volume", "standard deviation volume",
                          "size", "illiq", "capm beta", "max",
                          "capm alpha", "ivol", "turnover", "rvol", "bid-ask",
                          "detrended turnover", "standard deviation turnover", "rel to high",
                          "volume shock 30", "volume shock 60", "r2_1", "r7_2", "r13_2", "r22_2", "r31_2",
                          "r30_14", "r180_60", "var_5"]

        # Even though our data starts on 2017-01-01, we delete the first 180 days because we do not have data
        # there for a lot of the features we engineered ourselves
        self.startdate = pd.to_datetime("2017-01-01") + pd.to_timedelta(180, "day")
        # Last day where we have data for bitcoin
        self.enddate = pd.to_datetime("2022-03-01")
        self.test_percent = 0.5
        self.validation_percent = 1 / 3
        self.list_covar_hidden_neurons = list_covar_hidden_neurons
        self.list_factors_hidden_neurons = list_factors_hidden_neurons
        self.factor_dim = factor_dim
        self.within_time_limit = True
        self.predictiondays = predictiondays
        self.callback = [keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='auto',
            min_delta=0.0001,
            restore_best_weights=True,
            patience=300)]
        self.epochs = 1500
        self.tuning_trials = 24
        self.number_parallel_models = parallel_runs
        self.list_of_lists_contemporaneous_predictions = []
        self.list_of_lists_future_predictions = []
        self.weights_list = []
        self.full_input = full_input
        self.name = model_name + f" {factor_dim} factors"

    def initialize_X_Y(self, replace_nan_return_with=np.nan, replace_nan_covars_with=np.nan):
        regression_model.replace_infty(self)
        regression_model.prepare_rank_standardize_data(self)

        self.model_directory = "../Models/" + self.name

        Y = self.returns.iloc[1:].copy()
        # Y = Y.reset_index(drop=True)

        self.observed_mask = 1 - np.isnan(Y)

        Y = Y.replace(to_replace=np.nan, value=0)

        # Winsorize to remove extreme returns
        for i in np.arange(len(Y)):
            Y.iloc[i, :] = np.array(winsorize(Y.iloc[i, :], limits=[0, 0.05], nan_policy="omit"))

        X = [df.iloc[:-1] for df in self.variables_df_list]
        # X = [x.reset_index(drop=True) for x in X]
        X = [x.replace(to_replace=np.nan, value=replace_nan_covars_with) for x in X]

        # Add intercept
        for df in X:
            df["Intercept"] = 1

        self.X = X
        self.Y = Y
        self.initialize_Z()
        # self.calculate_sorted_portfolio_returns()
        self.populate_covar_sorted_portfolios()

        sorted_portfolios = self.sorted_portfolios

        # Split up in training and test set
        numberdays = Y.shape[0]
        testdays = int(np.ceil(numberdays * self.test_percent))
        traindays_end = numberdays - testdays
        traindays_start = 0
        validationdays = int(np.ceil(traindays_end * self.validation_percent))
        traindays_end = traindays_end - validationdays

        # Add constant regressor to the inputs
        Y["constant"] = 1
        sorted_portfolios["constant"] = 1

        Y_train, Y_valid, Y_test = Y.iloc[: traindays_end], Y.iloc[
                                                            traindays_end: traindays_end + validationdays], Y.iloc[
                                                                                                            traindays_end + validationdays:]

        portfolios_train, portfolios_valid, portfolios_test = \
            sorted_portfolios.iloc[: traindays_end], sorted_portfolios.iloc[
                                                     traindays_end: traindays_end + validationdays], sorted_portfolios.iloc[
                                                                                                     traindays_end + validationdays:]
        X_train, X_valid, X_test = [df.iloc[:traindays_end] for df in X], \
                                   [df.iloc[traindays_end: traindays_end + validationdays] for df in X], \
                                   [df.iloc[traindays_end + validationdays:] for df in X]

        # Remove the constant intercept from the outputs. We only needed them temporarily because in the inputs we do actually need the constant
        del Y["constant"]
        del Y_train["constant"]
        del Y_valid["constant"]
        del Y_test["constant"]

        if self.full_input:
            X_train.append(Y_train)
            X_valid.append(Y_valid)
            X_test.append(Y_test)
            X.append(Y)
        else:
            X_train.append(portfolios_train)
            X_valid.append(portfolios_valid)
            X_test.append(portfolios_test)
            X.append(sorted_portfolios)

        self.Y_test_global = Y_test
        self.X, self.Y, \
        self.X_train, self.Y_train, \
        self.X_train_global, self.Y_train_global, \
        self.X_valid, self.Y_valid, \
        self.X_valid_global, self.Y_valid_global, \
        self.X_test, self.Y_test, \
        self.X_test_global, self.Y_test_global, \
        self.traindays_start, self.traindays_end, self.validationdays, self.testdays = \
            X, Y, X_train, Y_train, X_train, Y_train, X_valid, Y_valid, \
            X_valid, Y_valid, X_test, Y_test, X_test, Y_test, traindays_start, traindays_end, validationdays, testdays

    # If we want to use the covariate-sorted portfolio, we need a matrix Z_t for every time point that contains all the
    # covariates of all coins at time t

    def initialize_Z(self):

        X = np.array(self.X)
        Z = X.swapaxes(0, 1)
        Z = [pd.DataFrame(Z[i, :, :]) for i in np.arange(len(Z))]
        for df in Z:
            df.columns = self.X[0].columns
            df.index = self.Y.columns

        Z_t = {day: df for day, df in zip(self.X[0].index, Z)}
        self.Z_t = Z_t
        self.reduce_Z_to_observed_values()

    # Reduces the matrices Z by eliminating the rows of coins that are not even observed on the day indexing the row
    def reduce_Z_to_observed_values(self):
        Y = self.Y
        Z_t = self.Z_t
        observed_coins = dict()

        for day in Y.index:
            series_of_returns_that_day = Y.loc[day, :]
            observed_coins[day] = [coin for coin in
                                   series_of_returns_that_day.index[~np.isnan(series_of_returns_that_day)]]

        # keep only days one day before returns
        shifted_days = [day - pd.to_timedelta(1, unit="day") for day in Y.index]

        Z_t_reduced = dict()
        for day_before in shifted_days:
            Z_t_reduced[day_before] = Z_t[day_before]

        for day_before, day in zip(shifted_days, Y.index):
            Z_t_reduced[day_before] = Z_t_reduced[day_before].loc[observed_coins[day], :]

        self.Z_t = Z_t_reduced
        self.observed_coins = observed_coins

    def calculate_sorted_portfolio_returns(self):
        Z_t = self.Z_t
        Y = self.Y

        portfolio_returns_per_day = {}

        for day_before, day in zip(Z_t.keys(), Y.index):
            portfolio_returns = np.matmul(np.transpose(Z_t[day_before].replace(np.nan, 0)),
                                          np.array(Y.loc[day, :].replace(np.nan, 0)))
            portfolio_returns_per_day[day] = portfolio_returns

        sorted_portfolios = pd.DataFrame.from_dict(portfolio_returns_per_day, orient="index")

        self.sorted_portfolios = sorted_portfolios

    def populate_covar_sorted_portfolios(self):
        Z_t = self.Z_t
        Y = self.Y
        observed_coins = self.observed_coins

        portfolios_per_day = {}

        for day_before, day in zip(Z_t.keys(), Y.index):
            ZtransposeZ = np.matmul(np.transpose(np.array(Z_t[day_before])), np.array(Z_t[day_before]))
            ZtransposeZ_inverse = np.linalg.inv(ZtransposeZ)
            Z_transpose_Z_inverse_Z = np.matmul(ZtransposeZ_inverse, np.transpose(np.array(Z_t[day_before])))
            portfolio_returns = np.matmul(Z_transpose_Z_inverse_Z, Y.loc[day, observed_coins[day]])
            portfolios_per_day[day] = portfolio_returns

        sorted_portfolios = pd.DataFrame.from_dict(portfolios_per_day, orient="index", columns=self.X[0].columns)

        self.sorted_portfolios = sorted_portfolios

    def shift_X_Y_train_valid_test_forwards(self):
        self.traindays_end = self.traindays_end + self.predictiondays
        self.traindays_start = self.traindays_start + self.predictiondays

        if self.testdays > self.predictiondays:
            self.testdays = self.testdays - self.predictiondays
            self.Y_train, self.Y_valid, self.Y_test = self.Y.iloc[
                                                      self.traindays_start: self.traindays_end], self.Y.iloc[
                                                                                                 self.traindays_end: self.traindays_end + self.validationdays], self.Y.iloc[
                                                                                                                                                                self.traindays_end + self.validationdays:]
            self.X_train, self.X_valid, self.X_test = [df.iloc[self.traindays_start:self.traindays_end] for df in
                                                       self.X], \
                                                      [df.iloc[
                                                       self.traindays_end: self.traindays_end + self.validationdays] for
                                                       df in self.X], \
                                                      [df.iloc[self.traindays_end + self.validationdays:] for df in
                                                       self.X]
        else:
            self.within_time_limit = False
            print("End of data frames reached.")

    def tune_hp(self):
        Y_train, X_train, Y_valid, X_valid, variables, factor_dim, list_covar_hidden_neurons = self.Y_train, self.X_train, self.Y_valid, self.X_valid, self.variables, self.factor_dim, self.list_covar_hidden_neurons
        hypermodel = ca_hypermodel(Y_train, X_train, Y_valid, X_valid, variables, factor_dim, list_covar_hidden_neurons)
        es = self.callback
        epochs = self.epochs
        trials = self.tuning_trials
        model_directory = self.model_directory

        tuner = BayesianOptimization(
            hypermodel,
            objective='val_loss',
            overwrite=True,
            max_trials=trials
        )
        tuner.search(X_train, Y_train, epochs=epochs, validation_data=(X_valid, Y_valid), callbacks=es)

        hyperparameters = tuner.get_best_hyperparameters()[0]

        number_parallel_models = self.number_parallel_models
        model = tuner.hypermodel.build(hyperparameters)
        model.save(model_directory + "/Hypermodel")
        '''for i, model in enumerate(model_list):
            random.seed(i)
            model.fit(
                x = X_train,
                y = Y_train,
                callbacks = es,
                epochs = epochs,
                validation_data = (X_valid, Y_valid)
            )'''

        # weights = [model.get_weights() for model in model_list]

        self.best_current_hp = hyperparameters
        # self.weights_list.append(weights)

    def load_hypermodel(self):
        directory = self.model_directory
        number_parallel = self.number_parallel_models
        model_list = [keras.models.load_model(directory + "/Hypermodel") for i in np.arange(number_parallel)]
        self.best_current_model_list = model_list

    def fit_model(self, stepnumber):
        print("Fitting step number " + str(stepnumber) + ".")

        X_train, X_valid, Y_train, Y_valid = self.X_train, self.X_valid, self.Y_train, self.Y_valid

        model_directory = self.model_directory

        es = self.callback
        epochs = self.epochs

        # Load hypermodel
        self.load_hypermodel()

        for i, model in enumerate(self.best_current_model_list):
            random.seed(i)
            training_history = model.fit(
                x=X_train,
                y=Y_train,
                callbacks=es,
                epochs=epochs,
                validation_data=(X_valid, Y_valid)
            )
            model.save(model_directory + "/" + str(stepnumber) + "/" + str(i))
            try:
                os.makedirs(model_directory + "/Training History/" + str(stepnumber))
            except:
                print("Directory " + model_directory + "/Training History/" + str(stepnumber) + " already exists.")
            with open(model_directory + "/Training History/" + str(stepnumber) + "/" + str(i) + ".pkl", "wb") as file:
                pickle.dump(training_history.history, file)

        # self.best_current_model_list = current_model_list

    def predict_next_testperiod_contemporaneous_returns(self):
        predictiondays = self.predictiondays
        X_test = self.X_test
        Y_test = self.Y_test

        current_model_list = self.best_current_model_list

        current_X_test = [df.iloc[:predictiondays] for df in X_test]
        current_Y_test = Y_test.iloc[:predictiondays]
        current_Y_pred_list = [model.predict(current_X_test) for model in current_model_list]
        current_Y_pred_list = [pd.DataFrame(current_Y_pred, index=current_Y_test.index, columns=current_Y_test.columns)
                               for current_Y_pred in current_Y_pred_list]

        self.list_of_lists_contemporaneous_predictions.append(current_Y_pred_list)

    def predict_contemporaneous_returns_from_saved(self, models, step, set_var_to_zero):
        predictiondays = self.predictiondays
        X_test = self.X_test
        Y_test = self.Y_test

        if set_var_to_zero != "none":
            for df in X_test[:-1]:
                df[set_var_to_zero].values[:] = 0

        current_model_list = models

        # Make sure that we have more enough days left to fill a prediction period. If not, shorten prediction period
        # Important for end of dataframe
        if predictiondays > Y_test.shape[0]:
            predictiondays = Y_test.shape[0]

        current_X_test = [df.iloc[:predictiondays] for df in X_test]
        current_Y_test = Y_test.iloc[:predictiondays]
        current_Y_pred_list = [model.predict(current_X_test) for model in current_model_list]
        current_Y_pred_list = [pd.DataFrame(current_Y_pred, index=current_Y_test.index, columns=current_Y_test.columns)
                               for current_Y_pred in current_Y_pred_list]

        for i, df in zip(np.arange(len(current_Y_pred_list)), current_Y_pred_list):
            if set_var_to_zero != "none":
                directory = self.model_directory + "/Contemporaneous Predictions Variable Importance/" + \
                            set_var_to_zero + "/Step " + str(step)
            else:
                directory = self.model_directory + "/Contemporaneous Predictions/Step " + str(step)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            df.to_csv(directory + "/Parallel Network " + str(i) + ".csv")

        # self.list_of_lists_contemporaneous_predictions.append(current_Y_pred_list)

    def predict_next_testperiod_future_returns(self):
        num_coins = self.Y_train.shape[1]
        predictiondays = self.predictiondays
        X_train = self.X_train
        X_valid = self.X_valid
        X_train = [pd.concat([train, valid]) for (train, valid) in zip(X_train, X_valid)]
        Y_train = self.Y_train
        Y_valid = self.Y_valid
        Y_train = pd.concat([Y_train, Y_valid])
        X_test = self.X_test
        Y_test = self.Y_test

        # Make sure that we have more enough days left to fill a prediction period. If not, shorten prediction period
        # Important for end of dataframe
        if predictiondays > Y_test.shape[0]: predictiondays = Y_test.shape[0]

        current_X_test = [df.iloc[:predictiondays] for df in X_test]
        current_Y_test = Y_test.iloc[:predictiondays]
        current_X = [pd.concat([train, test]) for (train, test) in zip(X_train, X_test)]

        # Model we work with
        model_list = self.best_current_model_list

        # Extract the (average) factor predictions from the model
        layer_name = 'Factors'
        factors_layer_model_list = [keras.Model(inputs=model.input,
                                                outputs=model.get_layer(layer_name).output) for model in model_list]
        factor_predictions_list = [model.predict(current_X) for model in factors_layer_model_list]
        factor_sample_averages_list = [np.array([factor_predictions[:i].mean(axis=0) for i in
                                                 np.arange(start=1, stop=factor_predictions.shape[0] + 1)]) for
                                       factor_predictions in factor_predictions_list]

        # Extract the beta predictions from the other part of the network
        layer_names = ["betas" + str(i) for i in np.arange(num_coins)]
        betas_layer_model_list = [[keras.Model(inputs=model.input,
                                               outputs=model.get_layer(layer_name).output) for
                                   layer_name in layer_names] for model in model_list]
        betas_predictions_list = [[model.predict(current_X) for model in parallel_model] for parallel_model in
                                  betas_layer_model_list]

        current_Y_future_pred_list = [np.array(
            [np.multiply(np.array(betas_predictions)[i][-predictiondays:, ],
                         factor_sample_averages[-(predictiondays + 1):-1, ]).sum(axis=1)
             for i in np.arange(len(betas_predictions))]).transpose()
                                      for (factor_sample_averages, betas_predictions) in
                                      zip(factor_sample_averages_list, betas_predictions_list)]

        current_Y_future_pred_list = [
            pd.DataFrame(current_Y_future_pred, index=current_Y_test.index, columns=current_Y_test.columns)
            for current_Y_future_pred in current_Y_future_pred_list]

        self.list_of_lists_future_predictions.append(current_Y_future_pred_list)

    def predict_future_returns_from_saved(self, models, step, factor_averaging_days="all", set_var_to_zero="none"):
        num_coins = self.Y_train.shape[1]
        predictiondays = self.predictiondays
        X_train = self.X_train

        X_valid = self.X_valid
        X_train = [pd.concat([train, valid]) for (train, valid) in zip(X_train, X_valid)]

        Y_train = self.Y_train
        Y_valid = self.Y_valid
        Y_train = pd.concat([Y_train, Y_valid])
        X_test = self.X_test
        Y_test = self.Y_test

        # Make sure that we have more enough days left to fill a prediction period. If not, shorten prediction period
        # Important for end of dataframe
        if predictiondays > Y_test.shape[0]: predictiondays = Y_test.shape[0]

        current_X_test = [df.iloc[:predictiondays] for df in X_test]
        current_Y_test = Y_test.iloc[:predictiondays]
        current_X = [pd.concat([train, test]) for (train, test) in zip(X_train, current_X_test)]

        # If we want to test variable importance
        if set_var_to_zero != "none":
            for df in current_X[:-1]:
                df[set_var_to_zero].values[:] = 0

        # Model we work with
        model_list = models

        # Extract the (average) factor predictions from the model
        layer_name = 'Factors'
        factors_layer_model_list = [keras.Model(inputs=model.input,
                                                outputs=model.get_layer(layer_name).output) for model in model_list]
        factor_predictions_list = [model.predict(current_X) for model in factors_layer_model_list]

        if factor_averaging_days == "all":
            factor_sample_averages_list = [
                np.array([factor_predictions[:i].mean(axis=0) for i in
                          np.arange(start=1, stop=factor_predictions.shape[0] + 1)]) for factor_predictions in
                factor_predictions_list]
        else:
            factor_sample_averages_list = [
                np.array([factor_predictions[np.max([i - factor_averaging_days, 0]):i].mean(axis=0) for i in
                          np.arange(start=1, stop=factor_predictions.shape[0] + 1)]) for factor_predictions in
                factor_predictions_list]

        if set_var_to_zero != "none":
            directory = self.model_directory + "/Betas and Factors Predictions Variable Importance/" + \
                        set_var_to_zero + "/Factors/Step " + str(step)
        else:
            directory = self.model_directory + "/Betas and Factors Predictions/Factors/Step " + str(step)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        for i, df in zip(np.arange(len(factor_predictions_list)), factor_predictions_list):
            with open(directory + "/" + str(i) + ".pkl", "wb") as file:
                pickle.dump(df, file)

        # Extract the beta predictions from the other part of the network
        layer_names = ["betas" + str(i) for i in np.arange(num_coins)]
        betas_layer_model_list = [keras.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(layer_name).output for layer_name in layer_names]
        )
            for model in model_list]
        betas_predictions_list = [model.predict(current_X) for model in betas_layer_model_list]

        if set_var_to_zero != "none":
            directory = self.model_directory + "/Betas and Factors Predictions Variable Importance/" + \
                        set_var_to_zero + "/Betas/Step " + str(step)
        else:
            directory = self.model_directory + "/Betas and Factors Predictions/Betas/Step " + str(step)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        for betas_predictions in betas_predictions_list:
            with open(directory + "/" + str(i) + ".pkl", "wb") as file:
                pickle.dump(betas_predictions, file)

        current_Y_future_pred_list = [np.array(
            [np.multiply(np.array(betas_predictions)[i][-predictiondays:, ],
                         factor_sample_averages[-(predictiondays + 1):-1, ]).sum(axis=1)
             for i in np.arange(len(betas_predictions))]).transpose()
                                      for (factor_sample_averages, betas_predictions) in
                                      zip(factor_sample_averages_list, betas_predictions_list)]

        current_Y_future_pred_list = [
            pd.DataFrame(current_Y_future_pred, index=current_Y_test.index, columns=current_Y_test.columns)
            for current_Y_future_pred in current_Y_future_pred_list]

        for i, df in zip(np.arange(len(current_Y_future_pred_list)), current_Y_future_pred_list):
            if set_var_to_zero == "none":
                if factor_averaging_days == "all":
                    directory = self.model_directory + "/Future Predictions/Step " + str(step)
                else:
                    directory = self.model_directory + "/Future Predictions " + \
                                str(factor_averaging_days) + " days factor average/Step " + str(step)
            else:
                if factor_averaging_days == "all":
                    directory = self.model_directory + "/Future Predictions Variable Importance/" + \
                                set_var_to_zero + "/Step " + str(step)
                else:
                    directory = self.model_directory + "/Future Predictions Variable Importance " + \
                                str(factor_averaging_days) + " days factor average/" \
                                + set_var_to_zero + "/Step " + str(step)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            df.to_csv(directory + "/Parallel Network " + str(i) + ".csv")

        # self.list_of_lists_future_predictions.append(current_Y_future_pred_list)

    def fit_all_models(self, step=0):
        stepnumber = 0

        while stepnumber < step:
            self.shift_X_Y_train_valid_test_forwards()
            stepnumber = stepnumber + 1
            print(stepnumber)

        while self.within_time_limit:
            self.fit_model(stepnumber)
            self.shift_X_Y_train_valid_test_forwards()
            stepnumber = stepnumber + 1
            print(stepnumber)

    def reset_X_Y(self):
        self.X_train = self.X_train_global
        self.X_valid = self.X_valid_global
        self.X_test = self.X_test_global

        self.Y_train = self.Y_train_global
        self.Y_valid = self.Y_valid_global
        self.Y_test = self.Y_test_global

        # Reset everything
        numberdays = self.Y.shape[0]
        test_percent = self.test_percent
        testdays = int(np.ceil(numberdays * test_percent))
        traindays_end = numberdays - testdays
        validation_percent = self.validation_percent
        validationdays = int(np.ceil(traindays_end * validation_percent))
        traindays_end = traindays_end - validationdays
        self.traindays_start, self.traindays_end, self.validationdays, self.testdays = 0, traindays_end, validationdays, testdays

        self.within_time_limit = True

    def load_parallel_models(self, stepnumber):
        model_directory = self.model_directory
        number_parallel = self.number_parallel_models

        current_model_list = [keras.models.load_model(model_directory + "/" + str(stepnumber) + "/" + str(i)) for i in
                              np.arange(number_parallel)]
        return current_model_list

    def make_contemporaneous_predictions(self, set_var_to_zero="none"):
        self.reset_X_Y()
        stepnumber = 0
        while self.within_time_limit:
            print("Make predictions of step number " + str(stepnumber))
            loaded_models = self.load_parallel_models(stepnumber)
            self.predict_contemporaneous_returns_from_saved(models=loaded_models, step=stepnumber,
                                                            set_var_to_zero=set_var_to_zero)
            self.shift_X_Y_train_valid_test_forwards()
            stepnumber = stepnumber + 1

    def make_future_predictions(self, set_var_to_zero="none", factor_averaging_days="all"):
        self.reset_X_Y()
        stepnumber = 0
        while self.within_time_limit:
            print("Make predictions of step number " + str(stepnumber))
            loaded_models = self.load_parallel_models(stepnumber)
            self.predict_future_returns_from_saved(models=loaded_models, step=stepnumber,
                                                   factor_averaging_days=factor_averaging_days,
                                                   set_var_to_zero=set_var_to_zero)
            self.shift_X_Y_train_valid_test_forwards()
            stepnumber = stepnumber + 1

    def update_contemporaneous_Y_predictions(self, set_var_to_zero="none"):
        if set_var_to_zero == "none":
            contemporaneous_predictions_directory = self.model_directory + "/Contemporaneous Predictions"
        else:
            contemporaneous_predictions_directory = self.model_directory + \
                                                    "/Contemporaneous Predictions Variable Importance/" + \
                                                    set_var_to_zero
        stepnumbers_model = os.listdir(contemporaneous_predictions_directory)
        try:
            stepnumbers_model = [int(step[5:]) for step in stepnumbers_model]
            stepnumbers_model.sort()
        except:
            raise ValueError("There seem to be file names in the model directory that cannot be converted to integers.")

        number_parallel = self.number_parallel_models

        list_of_lists_contemporaneous_predictions = [
            [
                pd.read_csv(
                    contemporaneous_predictions_directory + "/Step " + str(step) + "/Parallel Network " + str(
                        i) + ".csv", index_col="day"
                ) for i in np.arange(number_parallel)
            ] for step in stepnumbers_model
        ]
        self.Y_pred_cont_list = [
            pd.concat([parallel_predictions[i] for parallel_predictions in list_of_lists_contemporaneous_predictions],
                      axis=0)
            for i in np.arange(number_parallel)]

    def update_future_Y_predictions(self, factor_averaging_days="all", set_var_to_zero="none"):
        if set_var_to_zero == "none":
            if factor_averaging_days == "all":
                future_predictions_directory = self.model_directory + "/Future Predictions"
            else:
                future_predictions_directory = self.model_directory + "/Future Predictions " + str(
                    factor_averaging_days) + " days factor average"
        else:
            if factor_averaging_days == "all":
                future_predictions_directory = self.model_directory + "/Future Predictions Variable Importance /" + \
                                               set_var_to_zero
            else:
                future_predictions_directory = self.model_directory + "/Future Predictions Variable Importance " + \
                                               str(factor_averaging_days) + \
                                               " days factor average/" + set_var_to_zero

        stepnumbers_model = os.listdir(future_predictions_directory)
        try:
            stepnumbers_model = [int(step[5:]) for step in stepnumbers_model]
            stepnumbers_model.sort()
        except:
            raise ValueError("There seem to be file names in the model directory that cannot be converted to integers.")

        number_parallel = self.number_parallel_models

        list_of_lists_future_predictions = [
            [
                pd.read_csv(
                    future_predictions_directory + "/Step " + str(step) + "/Parallel Network " + str(
                        i) + ".csv", index_col="day"
                ) for i in np.arange(number_parallel)
            ] for step in stepnumbers_model
        ]
        self.Y_pred_future_list = [
            pd.concat([parallel_predictions[i] for parallel_predictions in list_of_lists_future_predictions],
                      axis=0)
            for i in np.arange(number_parallel)]

    def calculate_contemporaneous_r2(self, masked=True):
        Y_test = self.Y_test_global
        Y_pred_cont_list = self.Y_pred_cont_list

        mask = 1 - self.exclusion_mask
        mask = np.array(mask.iloc[-self.testdays:, :])

        if not masked:
            mask = mask ** 0

        Y_pred_cont_panel = np.array(Y_pred_cont_list)

        Y_pred_cont_average = Y_pred_cont_panel.mean(axis=0)
        masked_Y_pred_cont_average = Y_pred_cont_average * mask
        # Total R^2
        # For every single parallel model
        sum_squared_residuals_list = [(((np.array(Y_test) - np.array(Y_pred_cont)) ** 2)).sum().sum() for Y_pred_cont in
                                      Y_pred_cont_list]
        masked_sum_squared_residuals_list = [
            ((((np.array(Y_test) - np.array(Y_pred_cont)) * mask) ** 2) * mask).sum().sum() for Y_pred_cont in
            Y_pred_cont_list]

        sum_squared_returns = ((Y_test ** 2)).sum().sum()
        masked_sum_squared_returns = ((Y_test ** 2) * mask).sum().sum()
        Rsquared_total_list = [1 - sum_squared_residuals / sum_squared_returns
                               for sum_squared_residuals in sum_squared_residuals_list]
        masked_Rsquared_total_list = [1 - masked_sum_squared_residuals / masked_sum_squared_returns
                                      for masked_sum_squared_residuals in masked_sum_squared_residuals_list]
        # For the average
        sum_squared_residuals_average = (((np.array(Y_test) - np.array(Y_pred_cont_average)) ** 2)).sum().sum()
        masked_sum_squared_residuals_average = (
                    ((np.array(Y_test) - np.array(masked_Y_pred_cont_average)) ** 2) * mask).sum().sum()
        Rsquared_total_average = 1 - sum_squared_residuals_average / sum_squared_returns
        masked_Rsquared_total_average = 1 - masked_sum_squared_residuals_average / masked_sum_squared_returns

        self.masked_Y_test = Y_test * mask
        self.masked_Y_pred_cont_average = masked_Y_pred_cont_average
        self.sum_squared_residuals_list = sum_squared_residuals_list
        self.masked_sum_squared_residuals_list = masked_sum_squared_residuals_list
        self.sum_squared_returns = sum_squared_returns
        self.masked_sum_squared_returns = masked_sum_squared_returns
        self.masked_Rsquared_total_list = masked_Rsquared_total_list
        self.sum_squared_residuals_average = sum_squared_residuals_average
        self.masked_sum_squared_residuals_average = masked_sum_squared_residuals_average

        self.Y_pred_cont_average = Y_pred_cont_average
        self.Rsquared_total_list = Rsquared_total_list
        self.masked_Rsquared_total_list = masked_Rsquared_total_list
        self.masked_Rsquared_total_average = masked_Rsquared_total_average
        self.Rsquared_total_average = Rsquared_total_average

    def calculate_predictive_r2(self):
        Y_test = self.Y_test_global
        Y_pred_future_list = self.Y_pred_future_list

        # Masked: Only predictions where Y_test nonzero are counted. Y_test = 0 is assumed to only be the case when coins are eliminated
        mask = 1 - self.exclusion_mask
        mask = np.array(mask.iloc[-self.testdays:, :])

        masked_Y_pred_future_list = [Y_pred_future * mask for Y_pred_future in Y_pred_future_list]

        Y_pred_future_panel = np.array(Y_pred_future_list)
        masked_Y_pred_future_panel = np.array(masked_Y_pred_future_list)
        Y_pred_future_average = Y_pred_future_panel.mean(axis=0)
        masked_Y_pred_future_average = masked_Y_pred_future_panel.mean(axis=0)

        # Predictive R^2
        # For every single parallel model
        sum_squared_residuals_predictive_list = [((np.array(Y_test) - np.array(Y_pred_future)) ** 2).sum().sum()
                                                 for Y_pred_future in Y_pred_future_list]

        sum_squared_returns = (Y_test ** 2).sum().sum()

        Rsquared_predictive_list = [1 - sum_squared_residuals_predictive / sum_squared_returns
                                    for sum_squared_residuals_predictive in sum_squared_residuals_predictive_list]

        masked_sum_squared_residuals_predictive_list = [
            (((np.array(Y_test) - np.array(Y_pred_future)) * mask) ** 2).sum().sum()
            for Y_pred_future in Y_pred_future_list]

        masked_Y_test = Y_test * mask

        sum_squared_returns = (Y_test ** 2).sum().sum()

        masked_sum_squared_returns = ((np.array(Y_test) * mask) ** 2).sum().sum()

        masked_Rsquared_predictive_list = [1 - sum_squared_residuals_predictive / masked_sum_squared_returns
                                           for sum_squared_residuals_predictive in
                                           masked_sum_squared_residuals_predictive_list]

        # For the average
        sum_squared_residuals_average = ((np.array(Y_test) - np.array(Y_pred_future_average)) ** 2).sum().sum()
        masked_sum_squared_residuals_average = (
                    ((np.array(Y_test) - np.array(Y_pred_future_average)) * mask) ** 2).sum().sum()
        Rsquared_predictive_average = 1 - sum_squared_residuals_average / sum_squared_returns
        masked_Rsquared_predictive_average = 1 - masked_sum_squared_residuals_average / masked_sum_squared_returns

        self.masked_Y_test = masked_Y_test

        self.masked_Y_pred_future_list = masked_Y_pred_future_list
        self.sum_squared_returns = sum_squared_returns
        self.masked_sum_squared_returns = masked_sum_squared_returns
        self.Y_pred_future_average = Y_pred_future_average
        self.masked_Y_pred_future_average = masked_Y_pred_future_average
        self.Rsquared_predictive_list = Rsquared_predictive_list
        self.masked_Rsquared_predictive_list = masked_Rsquared_predictive_list
        self.Rsquared_predictive_average = Rsquared_predictive_average
        self.masked_Rsquared_predictive_average = masked_Rsquared_predictive_average


class ca_hypermodel(kt.HyperModel):
    def __init__(self, Y_train, X_train, Y_valid, X_valid, variables, dim_factors, list_hidden_neurons):
        super().__init__(self)
        self.Y_train, self.X_train, self.Y_valid, self.X_valid, self.variables, self.dim_factors, self.list_hidden_neurons = \
            Y_train, X_train, Y_valid, X_valid, variables, dim_factors, list_hidden_neurons

    def build(self, hp):
        # Build model
        conditional_autoencoder_model = link_both_parts(num_coins=self.Y_train.shape[1],
                                                        input_dimension=self.X_train[-1].shape[1],
                                                        num_covariates=len(self.variables) + 1,
                                                        encoding_dim=self.dim_factors,
                                                        list_no_hidden=self.list_hidden_neurons,
                                                        lambda_reg=hp.Choice("reg parameter lambda",
                                                                             [0.005, 0.001, 0.0001]))
        # Optimizer we use
        opt = keras.optimizers.Adam(
            learning_rate=hp.Choice("learning rate", [0.0001, 0.0005, 0.001])
        )

        # Compile model
        conditional_autoencoder_model.compile(
            loss="mean_squared_error",
            optimizer=opt
        )
        return conditional_autoencoder_model

    def fit(self, hp, model, *args, **kwargs):
        # Fit the Model
        return model.fit(
            *args,
            batch_size=hp.Choice("batch size", [32, 64]),
            **kwargs
        )
