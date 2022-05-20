# https://stackoverflow.com/questions/16024677/generate-correlated-data-in-python-3-3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from operator import itemgetter
import time
from copy import deepcopy
import os
from fiddler import FiddlerApi
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
#import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import sklearn
import sklearn.linear_model
import sklearn.feature_selection


def generate_gaussian_data(num_vars, mu=None, correlation_matrix=None,
                           num_samples=1000,
                           seed=1, show_plots=False):
    """
    param num_vars: int num of variables
    param mu: [optional] list of means of length num_vars. If None,
              the variables are set to have zero mean
    param correlation_matrix: [optional] the covariance matrix of size
        (num_vars, num_vars). If None, then variables are uncorrelated with
        each other
    param num_samples: [optional] number of samples. Defaults to 1000
    param seed: [optional] seed for reproducibility. Defaults to 1
    param show_plots: [optional] display plots of variables one vs another. Defaults to False

    returns: array of shape [num_samples, num_vars] with each column a different random variable
    """

    np.random.seed(seed)

    num_vars = int(num_vars)
    if not mu:
        mu = [0] * num_vars

    if correlation_matrix is None:
        correlation_matrix = np.eye(num_vars)

    data = np.random.multivariate_normal(mu, correlation_matrix,
                                         size=num_samples)

    if show_plots:
        sns.set(style="ticks")
        df = pd.DataFrame(columns=['Var_' + str(i) for i in range(num_vars)],
                          data=data)
        g = sns.PairGrid(df)
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter);

    return data


def generate_binomial_data(num_vars, probability_positive_class=[],
                           num_samples=1000, seed=1, show_plots=False):
    """
    param num_vars: int num of variables
    param probability_positive_class: list vector of probabilities of each variables positive class
                    of length num_vars. If empty, each variable has probability 0.5
    param num_samples: [optional] number of samples. Defaults to 1000
    param seed: [optional] seed for reproducibility. Defaults to 1
    param show_plots: [optional] display plots of variables one vs another. Defaults to False

    returns: array of shape [num_samples, num_vars] with each column a different random variable
    """
    np.random.seed(seed)

    if not probability_positive_class:
        probability_positive_class = [0.5] * num_vars

    data = np.random.binomial(1, probability_positive_class,
                              (num_samples, num_vars))

    if show_plots:
        sns.set(style="ticks")
        df = pd.DataFrame(columns=['Var_' + str(i) for i in range(num_vars)],
                          data=data)
        g = sns.PairGrid(df)
        g.map_diag(plt.hist)
        g.map_offdiag(sns.barplot)

    return data


def generate_multinomial_data(num_classes, class_probabilities=[],
                              num_samples=1000, seed=1, show_plots=False):
    """
    param num_classes: int num of classes
    param class_probabilities: list vector of probabilities of each class
                    of length num_classes. If empty, each variable has probability 1/num_classes
    param num_samples: [optional] number of samples. Defaults to 1000
    param seed: [optional] seed for reproducibility. Defaults to 1
    param show_plots: [optional] display plots of variables one vs another. Defaults to False

    returns: array of shape [num_samples, num_vars] with each column a different random variable
    """
    np.random.seed(seed)

    if not class_probabilities:
        class_probabilities = [1 / num_classes] * num_classes

    # print(class_probabilities)
    raw_data = np.random.multinomial(1, class_probabilities, size=num_samples)
    data = np.argmax(raw_data, axis=1)

    if show_plots:
        unique, counts = np.unique(data, return_counts=True)
        sns.barplot(unique, counts)

    return data


def create_time_series(start_time, end_time=None, period_in_secs=None,
                       num_samples=1000):
    """
    param start_time: str containing the start timestamp in the 'yyyy-mm-dd hh:mm:ss'. For example
                      '2020-01-16 00:00:00'
    param end_time: [optional] str containing the start timestamp in the 'yyyy-mm-dd hh:mm:ss'. For example
                    '2020-01-16 00:00:00'. If end_time and period_in_secs are both provided, then num_samples
                    will be overridden
    param period_in_secs:[optional] time gap between each event. A default value of one sec will be used if endtime
                    is also None
    param num_samples:[optional] number of samples to be created.

    returns: Pandas Series of timestamps

    """
    time_mapping = {}


def drift_bounds(drift_period, data_len):
    if isinstance(drift_period[0], int) and isinstance(drift_period[1], int):
        drift_start_index = drift_period[0]
        drift_end_index = drift_period[1]
    elif isinstance(drift_period[0], float) and isinstance(drift_period[1],
                                                           float):
        drift_start_index = int(drift_period[0] * data_len)
        drift_end_index = int(drift_period[1] * data_len)

    assert drift_start_index >= 0
    assert drift_end_index <= data_len
    assert drift_end_index > drift_start_index

    return drift_start_index, drift_end_index


def drifted_numeric_data(dataset, num_drift_features, num_rows,
                         covariance_matrix, seed=1, show_plots=False):
    cols = num_drift_features.keys()
    num_vars = len(cols)
    mean_list = []
    var_list = []

    for col, shift in num_drift_features.items():
        data_col = dataset[col]
        mean = np.mean(data_col)
        var = np.var(data_col)
        new_mean = mean + shift['mean']
        new_var = var + shift['var']
        mean_list.append(new_mean)
        var_list.append(new_var)

    if not covariance_matrix:
        covariance_matrix = np.diag(var_list)
    print(covariance_matrix)
    data = generate_gaussian_data(num_vars,
                                  mu=mean_list,
                                  correlation_matrix=covariance_matrix,
                                  num_samples=num_rows,
                                  seed=seed,
                                  show_plots=show_plots)

    data_dict = {col: data[:, i] for i, col in enumerate(cols)}
    return data_dict


def drifted_categorical_data(cat_drift_features, num_rows, seed=1,
                             show_plots=False):
    cols = cat_drift_features.keys()
    num_vars = len(cols)
    data_dict = {}

    for col, probs in cat_drift_features.items():
        num_classes = len(probs.keys())
        class_probabilities = list(probs.values())
        data_dict[col] = generate_multinomial_data(num_classes,
                                                   class_probabilities,
                                                   num_samples=num_rows,
                                                   seed=seed,
                                                   show_plots=show_plots)

    return data_dict


def drifted_binary_data(bin_drift_features, num_rows, seed=1,
                        show_plots=False):
    cols = bin_drift_features.keys()
    num_vars = len(cols)
    data_dict = {}
    class_probabilities = []

    for col, probs in bin_drift_features.items():
        class_probabilities.append(probs)

    data = generate_binomial_data(num_vars,
                                  class_probabilities,
                                  num_samples=num_rows,
                                  seed=seed,
                                  show_plots=show_plots)

    data_dict = {col: data[:, i] for i, col in enumerate(cols)}

    return data_dict


def stitch_drift_data(dataset, drift_start_index, drift_end_index,
                      num_data_dict, cat_data_dict, bin_data_dict,
                      allow_outliers=False):
    if num_data_dict:
        for col in num_data_dict.keys():
            min_bound = np.min(dataset[col])
            max_bound = np.max(dataset[col])
            dataset[col].iloc[drift_start_index:drift_end_index] = \
            num_data_dict[col]
            if not allow_outliers:
                dataset[col] = dataset[col].apply(
                    lambda x: min_bound if x < min_bound else \
                        (max_bound if x > max_bound else x))

    if cat_data_dict:
        for col in cat_data_dict.keys():
            mapping = {i: k for i, k in
                       enumerate(sorted(dataset[col].unique()))}
            dataset[col].iloc[drift_start_index:drift_end_index] = \
            cat_data_dict[col]
            # print(dataset[col].iloc[drift_start_index:drift_end_index])
            # print(sorted(dataset[col].unique()))
            dataset[col].iloc[drift_start_index:drift_end_index] = \
                dataset[col].iloc[drift_start_index:drift_end_index].map(
                    mapping)

    if bin_data_dict:
        for col in bin_data_dict.keys():
            mapping = {i: k for i, k in
                       enumerate(sorted(dataset[col].unique()))}
            dataset[col].iloc[drift_start_index:drift_end_index] = \
            bin_data_dict[col]
            dataset[col].iloc[drift_start_index:drift_end_index] = \
                dataset[col].iloc[drift_start_index:drift_end_index].map(
                    mapping)

    return dataset


def induce_drift_through_data(dataset,
                              drift_period,
                              num_drift_features={},
                              bin_drift_features={},
                              cat_drift_features={},
                              covariance_matrix=None,
                              seed=1,
                              show_plots=False):
    """
    Induce drift in given dataset for the given drift period.

    :param dataset: Pandas df
    :param num_drift_features: dict containing a mapping of the numeric drift features and a dict containing the mean
                               and variance shift for each e.g. {'feature': {'mean': 1, 'var': 0}}
    :param bin_drift_features: dict containing a mapping of the binary drift features and the positive class prob
                               {'feature_1': 0.7, 'feature_2': 0.8}
    :param cat_drift_features: dict containing a mapping of the categorical drift features and a dict containing the
                               new probabilities for each label e.g.
                               {'feature': {'label1': 0.7, 'label2': 0.2, 'label3': 0.1}}
    :drift_period: list containing two integers, the start and end row indices of the drift period OR two floats,
                               representing the start and end of the fraction of data to be drifted
    :covariance_matrix: [optional] covariance between the numerical features. By default they'll be independent
    :seed: [optional] random seed
    :show_plots: [optional] show plots of the final data and the intermediate processes

    :returns drifted_dataset: a copy of the dataset with drift induced
    """

    drift_dataset = dataset.copy(deep=True)
    data_len = len(dataset)

    drift_start_index, drift_end_index = drift_bounds(drift_period, data_len)
    num_rows_to_change = drift_end_index - drift_start_index

    if not (num_drift_features or cat_drift_features or bin_drift_features):
        raise ValueError('No features specified for drift')

    num_data_dict, cat_data_dict, bin_data_dict = {}, {}, {}

    if num_drift_features:
        num_data_dict = drifted_numeric_data(dataset, num_drift_features,
                                             num_rows_to_change,
                                             covariance_matrix,
                                             seed=seed, show_plots=show_plots)

    if cat_drift_features:
        cat_data_dict = drifted_categorical_data(cat_drift_features,
                                                 num_rows_to_change,
                                                 seed=seed,
                                                 show_plots=show_plots)

    if bin_drift_features:
        bin_data_dict = drifted_binary_data(bin_drift_features,
                                            num_rows_to_change,
                                            seed=seed, show_plots=show_plots)

    drift_dataset = stitch_drift_data(dataset, drift_start_index,
                                      drift_end_index,
                                      num_data_dict, cat_data_dict,
                                      bin_data_dict)

    if show_plots:
        g = sns.PairGrid(drift_dataset)
        g.map_diag(plt.hist)
        g.map_offdiag(sns.barplot)

    return drift_dataset


# Function to sample 'n' variables from cols
def sample_n_columns(cols, n):
    t = 1000 * time.time()
    np.random.seed(int(t) % 2 ** 32)
    max_num = len(cols)
    samples = np.random.choice(range(max_num), size=n, replace=False)
    return [cols[i] for i in samples]


def get_num_drift_features_deltas(dataset, features, time_period, period_start,
                                  drift_tracker):
    if not features:
        return {}
    num_drift_features = {}
    if time_period not in drift_tracker.keys():
        drift_tracker[time_period] = {}
    if period_start not in drift_tracker[time_period]:
        drift_tracker[time_period][period_start] = {}

    feature_stats = {}
    for feature in features:
        mean = np.mean(dataset[feature])
        var = np.var(dataset[feature])
        mean_change = (2 * np.random.random() - 1) * mean
        var_change = (2 * np.random.random() - 1) * var
        num_drift_features[feature] = {'mean': mean_change, 'var': var_change}
        feature_stats[feature] = {'original_mean': mean,
                                  'original_var': var,
                                  'new_mean_intended': mean + mean_change,
                                  'new_var_intended': var + var_change}

    drift_tracker[time_period][period_start]['num_features'] = feature_stats
    return num_drift_features


def get_cat_drift_features_probs(dataset, features, time_period, period_start,
                                 drift_tracker):
    if not features:
        return {}
    cat_drift_features = {}
    if time_period not in drift_tracker.keys():
        drift_tracker[time_period] = {}
    if period_start not in drift_tracker[time_period]:
        drift_tracker[time_period][period_start] = {}

    feature_probs = {}
    cat_drift_features = {}
    for feature in features:
        probs = dataset.groupby(feature).size().div(len(dataset))
        new_probs = np.random.choice(100, len(probs))
        new_probs = new_probs / np.sum(new_probs)
        cat_drift_features[feature] = {k: new_probs[i] for i, k in
                                       enumerate(sorted(list(probs.keys())))}

        feature_probs[feature] = {'original_probs': probs.to_dict(),
                                  'new_probs_intended': cat_drift_features[
                                      feature]}

    drift_tracker[time_period][period_start]['cat_features'] = feature_probs

    return cat_drift_features


def get_bin_drift_features_probs(dataset, features, time_period, period_start,
                                 drift_tracker):
    if not features:
        return {}
    cat_drift_features = {}
    if time_period not in drift_tracker.keys():
        drift_tracker[time_period] = {}

    if period_start not in drift_tracker[time_period]:
        drift_tracker[time_period][period_start] = {}
    feature_probs = {}
    bin_drift_features = {}
    for feature in features:
        probs = dataset.groupby(feature).size().div(len(dataset))
        new_probs = np.random.choice(100, len(probs))
        new_probs = new_probs / np.sum(new_probs)

        bin_drift_features[feature] = new_probs[1]
        feature_probs[feature] = {'original_probs': probs.to_dict(),
                                  'new_probs_intended': {
                                  k: new_probs.tolist()[i] for i, k in
                                  enumerate(probs.to_dict().keys())}}

    drift_tracker[time_period][period_start]['bin_features'] = feature_probs

    return bin_drift_features


def parse_and_add_drift_tracker(dataset, drift_tracker):
    time_periods = drift_tracker.keys()
    for time_period in time_periods:
        start_times = drift_tracker[time_period].keys()
        for start_time in start_times:
            change_dict = drift_tracker[time_period][start_time]
            feature_dtypes = change_dict.keys()
            for dtype in feature_dtypes:
                if dtype == 'num_features':
                    for feature in change_dict[dtype]:
                        new_mean = np.mean(dataset[feature][
                                           start_time: start_time + time_period])
                        new_var = np.var(dataset[feature][
                                         start_time: start_time + time_period])
                        change_dict[dtype][feature]['new_mean'] = new_mean
                        change_dict[dtype][feature]['new_var'] = new_var

                else:
                    for feature in change_dict[dtype]:
                        new_probs = dataset[
                                    start_time: start_time + time_period] \
                            .groupby(feature).size().div(time_period).to_dict()
                        change_dict[dtype][feature]['new_probs'] = new_probs
