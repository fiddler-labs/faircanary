import numpy as np
import pandas as pd
from fiddler import FiddlerApi
import os
from sklearn.model_selection import train_test_split
import sklearn.pipeline
import category_encoders
import sklearn.linear_model
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize, scale
from sklearn.preprocessing import MaxAbsScaler
from scipy import stats
from scipy.spatial import distance
import itertools


# The current implementation, one feature at a time
def adversarial_linear_drift_score(original_df, target_df, feature_list=None):
    """Calculates AUC value for a logistic regression designed to
       distinguish between the prior and post datasets. Minor changes from Fiddler code"""
    assert (original_df.columns == target_df.columns).all()
    if not feature_list:
        feature_list = original_df.columns
    gini = {}
    for feature in feature_list:
        # null values are dropped
        x_orig = original_df[feature].dropna()
        x_target = target_df[feature].dropna()

        orig_len, new_len = len(x_orig), len(x_target)
        if orig_len <= new_len:
            x_target = x_target.sample(orig_len, random_state=1)
        else:
            x_orig = x_orig.sample(new_len, random_state=1)

        if len(x_orig) > 200:  # sufficiency of data to detect drift
            x_orig, x_target = x_orig.to_frame(), x_target.to_frame()
            x_orig['label'], x_target['label'] = 0, 1
            data = pd.concat(
                [x_orig[[feature, 'label']], x_target[[feature, 'label']]])
            x = data[feature].to_frame()
            y = data['label'].to_frame()

            logistic_regression = sklearn.pipeline.Pipeline([
                ('one_hot', category_encoders.OneHotEncoder(
                    cols=x.select_dtypes('object').columns.tolist(),
                    handle_unknown='ignore', use_cat_names=True)),
                ('lr',
                 sklearn.linear_model.LogisticRegression(solver='lbfgs'))
            ])
            clf = logistic_regression.fit(x, y)
            predictions = clf.predict_proba(x)
            gini_score = 2 * roc_auc_score(y, predictions[:, 1]) - 1
            gini_score = max(gini_score, 0)
            gini[feature] = gini_score

    return gini


def adversarial_quadratic_drift_score(original_df, target_df,
                                      feature_list=None):
    """Calculates AUC value for a logistic regression designed to
       distinguish between the prior and post datasets. Minor changes from Fiddler code"""
    assert (original_df.columns == target_df.columns).all()
    if not feature_list:
        feature_list = original_df.columns
    gini = {}
    for feature in feature_list:
        # null values are dropped
        x_orig = original_df[feature].dropna()
        x_target = target_df[feature].dropna()

        orig_len, new_len = len(x_orig), len(x_target)
        if orig_len <= new_len:
            x_target = x_target.sample(orig_len, random_state=1)
        else:
            x_orig = x_orig.sample(new_len, random_state=1)

        if len(x_orig) > 200:  # sufficiency of data to detect drift
            x_orig, x_target = x_orig.to_frame(), x_target.to_frame()
            x_orig['label'], x_target['label'] = 0, 1
            data = pd.concat(
                [x_orig[[feature, 'label']], x_target[[feature, 'label']]])
            x = data[feature].to_frame()
            y = data['label'].to_frame()

            logistic_regression = sklearn.pipeline.Pipeline([
                ('one_hot', category_encoders.OneHotEncoder(
                    cols=x.select_dtypes('object').columns.tolist(),
                    handle_unknown='ignore', use_cat_names=True)),
                ('lr',
                 sklearn.linear_model.LogisticRegression(solver='lbfgs'))
            ])
            x = x + x ** 2
            clf = logistic_regression.fit(x, y)
            predictions = clf.predict_proba(x)
            gini_score = 2 * roc_auc_score(y, predictions[:, 1]) - 1
            gini_score = max(gini_score, 0)
            gini[feature] = gini_score

    return gini


# Building a logistic regression with all features at once
def detect_drift_all_features(original_df, target_df):
    """Calculates AUC value for a logistic regression designed to
       distinguish between the prior and post datasets"""
    assert (original_df.columns == target_df.columns).all()
    gini = {}
    x_orig = original_df.copy()
    x_target = target_df.copy()
    orig_len, new_len = len(x_orig), len(x_target)
    if orig_len <= new_len:
        x_target = x_target.sample(orig_len, random_state=1)
    else:
        x_orig = x_orig.sample(new_len, random_state=1)

    if len(x_orig) > 200:  # sufficiency of data to detect drift
        # x_orig, x_target = x_orig.to_frame(), x_target.to_frame()
        x_orig['label'], x_target['label'] = 0, 1
        data = pd.concat([x_orig, x_target])
        x = data.loc[:, data.columns != 'label']
        y = data['label'].to_frame()
        logistic_regression = sklearn.pipeline.Pipeline([
            ('one_hot', category_encoders.OneHotEncoder(
                cols=x.select_dtypes('object').columns.tolist(),
                handle_unknown='ignore', use_cat_names=True)),
            ('lr',
             sklearn.linear_model.LogisticRegression(fit_intercept=False,
                                                     C=1e9))
        ])
        clf = logistic_regression.fit(x, y)
        predictions = clf.predict_proba(x)
        gini_score = 2 * roc_auc_score(y, predictions[:, 1]) - 1
        gini_score = max(gini_score, 0)
        gini['score'] = gini_score

    return gini, clf


def ks_drift_score(original_df, target_df):
    ks = {}
    assert (original_df.columns == target_df.columns).all()

    orig_len, tar_len = len(original_df), len(target_df)
    if orig_len < tar_len:
        target_df = target_df.sample(orig_len, random_state=1)
    elif orig_len > tar_len:
        original_df = original_df.sample(tar_len, random_state=1)

    for col in original_df.columns:
        ks_stat_col = stats.ks_2samp(original_df[col], target_df[col])
        # ks[col] = {'statistic': ks_stat_col.statistic,
        #           'pvalue': ks_stat_col.pvalue}
        ks[col] = ks_stat_col.statistic

    return ks


def jsd_drift_score(original_df, target_df, mapping_dict={}):
    jsd = {}
    assert (original_df.columns == target_df.columns).all()

    orig_len, tar_len = len(original_df), len(target_df)
    if orig_len < tar_len:
        target_df = target_df.sample(orig_len, random_state=1)
    elif orig_len > tar_len:
        original_df = original_df.sample(tar_len, random_state=1)

    for col in original_df.columns:
        org_data_col = original_df[col]
        tar_data_col = target_df[col]
        if org_data_col.dtype not in ['int64', 'float64']:
            # unique_vals = org_data_col.unique()
            # mapping_dict = {k:i for i,k in enumerate(sorted(unique_vals))}
            org_data_col = org_data_col.map(mapping_dict[col])
            tar_data_col = tar_data_col.map(mapping_dict[col])

        min_org, max_org = min(org_data_col), max(org_data_col)
        min_tar, max_tar = min(tar_data_col), max(tar_data_col)

        bin_min, bin_max = min(min_org, min_tar), max(max_org, max_tar)
        bin_width = (bin_max - bin_min) / 20
        bin_max = bin_max + bin_width
        hist_org, _ = np.histogram(org_data_col,
                                   bins=np.arange(bin_min, bin_max, bin_width),
                                   normed=1)
        hist_tar, _ = np.histogram(tar_data_col,
                                   bins=np.arange(bin_min, bin_max, bin_width),
                                   normed=1)

        jsd_col = distance.jensenshannon(hist_org, hist_tar)
        jsd[col] = jsd_col
    return jsd


def plot_num_data_distribution(org_df, tar_df, cols=None):
    plot_df_org = org_df.copy(deep=True)
    plot_df_tar = tar_df.copy(deep=True)
    plot_df_org['Label'] = 'Original'
    plot_df_tar['Label'] = 'Target'

    if not cols:
        cols = org_df.columns.tolist()
        assert cols == tar_df.columns.tolist(
        )
    plot_df = pd.concat((plot_df_org, plot_df_tar))
    g = sns.PairGrid(plot_df, vars=cols, hue='Label')
    g.map_diag(plt.hist)
    g.map_lower(sns.scatterplot)


def plot_cat_data_distribution(org_df, tar_df, cols=None):
    plot_df_org = org_df.copy(deep=True)
    plot_df_tar = tar_df.copy(deep=True)
    plot_df_org['Label'] = 'Original'
    plot_df_tar['Label'] = 'Target'

    if not cols:
        cols = org_df.columns.tolist()
        assert cols == tar_df.columns.tolist(
        )
    plot_df = pd.concat((plot_df_org, plot_df_tar))
    fig, axs = plt.subplots(nrows=len(cols), figsize=(6, 6 * len(cols)))

    for i, col in enumerate(cols):
        sns.countplot(x=col, data=plot_df, hue='Label', ax=axs[i])


def plot_data_drift(original_df, target_df, drift_algo_function_mapping,
                    drift_algorithms=['ks'], mapping_dict={}):
    num_plots = len(drift_algorithms)
    plot_size_y = 12 * np.ceil(num_plots / 2)
    plot_size_x = 6 * min(2, num_plots)
    fig, axs = plt.subplots(ncols=num_plots,
                            figsize=(plot_size_y, plot_size_x))

    for index, algo in enumerate(drift_algorithms):
        if algo == 'jsd':
            drift_score = drift_algo_function_mapping[algo](original_df,
                                                            target_df,
                                                            mapping_dict)
        else:
            drift_score = drift_algo_function_mapping[algo](original_df,
                                                            target_df)
        drift_df = pd.DataFrame([drift_score]).T.reset_index() \
            .rename({'index': 'features', 0: 'drift_score'}, axis='columns') \
            .sort_values(by='drift_score', ascending=False)
        sns.barplot(y='features', x='drift_score', data=drift_df,
                    ax=axs[index]).set_title(f'Drift Score for {algo}')

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.5, hspace=0.5)
