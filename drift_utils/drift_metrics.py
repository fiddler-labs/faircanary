import pandas as pd
import numpy as np
import torch
# https://stackoverflow.com/questions/16024677/generate-correlated-data-in-python-3-3
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from operator import itemgetter
import time
from copy import deepcopy
import os
from fiddler import FiddlerApi
from sklearn.preprocessing import MaxAbsScaler
#import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import sklearn
import sklearn.linear_model
import sklearn.feature_selection
from scipy import stats
from scipy.spatial import distance
from sklearn.utils.extmath import cartesian
from copy import deepcopy
from scipy.stats import wasserstein_distance


def get_jsd(reference, target):
    min_org, max_org = min(reference), max(reference)
    min_tar, max_tar = min(target), max(target)

    bin_min, bin_max = min(min_org, min_tar), max(max_org, max_tar)
    bin_width = (bin_max - bin_min) / 20
    bin_max = bin_max + bin_width
    hist_org, _ = np.histogram(reference,
                               bins=np.arange(bin_min, bin_max, bin_width),
                               normed=True)
    hist_tar, _ = np.histogram(target,
                               bins=np.arange(bin_min, bin_max, bin_width),
                               normed=True)
    return distance.jensenshannon(hist_org, hist_tar)


def get_ads(reference, target, num_quantiles=10):
    ads = 0
    points_per_bucket = len(reference) // num_quantiles
    for bucket in range(num_quantiles):
        step = points_per_bucket * bucket
        bucket_ref_mean = np.mean(reference[step:step + points_per_bucket])
        bucket_tar_mean = np.mean(target[step:step + points_per_bucket])
        ads += np.abs(bucket_ref_mean - bucket_tar_mean) / num_quantiles
    return ads


def ks_drift_score(targets, reference):
    """

    :param reference: numpy array
    :param targets: numpy array
    :return: ks statistic
    """

    ks_scores = []
    if reference.shape == targets.shape:
        return stats.ks_2samp(targets, reference).statistic
    else:
        for target in targets:
            ks_scores.append(stats.ks_2samp(target, reference).statistic)
    return ks_scores


def jsd_drift_score(targets, reference):
    """

    :param reference: numpy array
    :param targets: numpy array
    :return: Jensen Shannon distance
    """

    if reference.shape == targets.shape:
        return get_jsd(targets, reference)

    else:
        jsd_scores = []

        for target in targets:
            jsd_scores.append(get_jsd(target, reference))

    return jsd_scores


def aggregate_drift_score(targets, reference, num_quantiles=10):
    """

    :param reference: numpy array
    :param targets: numpy array
    :param num_quantiles: num of quantile buckets
    :return: aggregate drift score
    """

    if reference.shape == targets.shape:
        return get_ads(targets, reference, num_quantiles=num_quantiles)
    else:
        aggregate_drift_scores = []
        for target in targets:
            aggregate_drift_scores.append(get_ads(target, reference,
                                                  num_quantiles=num_quantiles))

        return aggregate_drift_scores


def wasserstein_drift_score(targets, reference):
    wd = []
    #print(reference.shape, targets.shape)
    if reference.shape == targets.shape:
        return wasserstein_distance(targets, reference)
    else:
        for target in targets:
            #print('inner loop', reference, target)
            wd.append(wasserstein_distance(target, reference))
    return wd


def wasserstein_drift_score_torch(targets, reference):
    wd = []
    reference.sort()
    #print(reference.shape, targets.shape)
    if reference.shape == targets.shape:
        targets.sort()
        return torch.mean(torch.abs(targets - reference))
    else:
        for target in targets:
            target.sort()
            wd.append(torch.mean(torch.abs(target - reference)))
    return wd



def mean_drift_score(targets, reference):
    means = []
    #print(reference.shape, targets.shape)
    if reference.shape == targets.shape:
        return np.mean(targets, axis=-1) - np.mean(reference, axis=-1)
    else:
        for target in targets:
            #print(reference, target)
            means.append(np.mean(target, axis=-1) -
                         np.mean(reference, axis=-1))
        return means


def drift_score(target, reference, score='mean'):

    if score == 'mean':
        return np.mean(target, axis=-1) - np.mean(reference, axis=-1)

    if score == 'jsd':
        return jsd_drift_score(target, reference)

    if score == 'ks':
        return ks_drift_score(target, reference)

    if score == 'wd':
        return wasserstein_drift_score(target, reference)

    if score == 'ads':
        return aggregate_drift_score(target, reference)

    return 0
