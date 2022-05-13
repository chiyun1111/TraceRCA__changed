"""run_selecting_features.py

selecting useful features for each fault
========================================

We determine whether a feature is useful by testing whether
the distribution of normal and abnormal invocations with
respect to it changes after the fault occurs.

Feature candidate set
---------------------
In a microservice system, there are various metrics. In Train-Ticket data, we use
latency and HTTP status of each invocation, and CPU usage,
memory usage, network receive/send throughput, and disk
read/write throughput of each microservice as the features for
trace anomaly detection.

Note that we only consider the historical invocations of the same microservice
pair  to which this invocation belongs because the
underlying distributions with respect to the same feature can
vary vastly for different microservice pairs.


Parameters
  1. Input_file : the data after the fault happens (pkl)
  2. History : all the historical  invocations of the same microservice pair
     1) in the last slot and 2) in the same slot of the last period (pkl)
  3. output_file : the useful features of each invocation (dict)
  4. fisher_threshold : a given threshold to test whether the feature of the invocation is useful


"""


import pickle
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm
from pprint import pprint
# from trainticket_config import FEATURE_NAMES
#自定义所要使用的feature
FEATURE_NAMES = ['latency','http_status','cpu_use','mem_use_percent','mem_use_amount','file_write_rate','file_read_rate',
                 'net_send_rate','net_receive_rate']

DEBUG = False  # very slow


def distribution_criteria(empirical, reference, threshold):
    """This function has not been used
    """
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    ref_ratio = sum(np.abs(reference - historical_mean) > 3 * historical_std) / reference.shape[0]
    emp_ratio = sum(np.abs(empirical - historical_mean) > 3 * historical_std) / empirical.shape[0]
    return (emp_ratio - ref_ratio) > threshold * ref_ratio


def fisher_criteria(empirical, reference, side='two-sided'):
    """This function has not been used
    """
    if side == 'two-sided':
        diff_mean = (np.abs(np.mean(empirical) - np.mean(reference)) ** 2)
    elif side == 'less':
        diff_mean = np.maximum(np.mean(empirical) - np.mean(reference), 0) ** 2
    elif side == 'greater':
        diff_mean = np.maximum(np.mean(reference) - np.mean(empirical), 0) ** 2
    else:
        raise RuntimeError(f'invalid side: {side}')
    variance = np.maximum(np.var(empirical) + np.var(reference), 0.1)
    return diff_mean / variance


def stderr_criteria(empirical, reference, threshold):
    """Testing whether the feature of the invocation is useful

    :param empirical: the data after the fault happens
    :param reference: the normal data before the fault happens (contains twofold stage)
    :param threshold: a given threshold to test whether the feature of the invocation is useful
    :return: bool type
    """
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


# @click.command('invocation feature selection')
# @click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
# @click.option('-o', '--output', 'output_file', default='.', type=str)
# @click.option('-h', '--history', default='historical_data.pkl', type=str)
# @click.option("-f", "--fisher", "fisher_threshold", default=1, type=float)



def selecting_feature_main(input_file: str, output_file: str, history: str, fisher_threshold):
    """The main function to select the useful features

    :param input_file: the data after the fault happens (pkl)
    :param output_file: the useful features of each invocation (dict)
    :param history: the normal data before the fault happens (contains twofold stage)
    :param fisher_threshold: a given threshold to test whether the feature of the invocation is useful
    :return:
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    with open(history, 'rb') as f:
        history = pickle.load(f)
    # logger.debug(f'{input_file}')
    with open(str(input_file), 'rb') as f:
        df = pickle.load(f)
    df = df.set_index(keys=['source', 'target'], drop=True).sort_index()
    df['http_status'] = pd.to_numeric(df['http_status'])
    history['http_status'] = pd.to_numeric(history['http_status'])
    history = history.set_index(keys=['source', 'target'], drop=True).sort_index()
    indices = np.intersect1d(np.unique(df.index.values), np.unique(history.index.values))
    useful_features_dict = defaultdict(list)
    if DEBUG:
        plot_dir = output_file.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)
    for (source, target), feature in tqdm(product(indices, FEATURE_NAMES)): #笛卡尔积
        empirical = np.sort(df.loc[(source, target), feature].values)
        reference = np.sort(history.loc[(source, target), feature].values)
        # p_value = ks_2samp(
        #     empirical, reference, alternative=ALTERNATIVE[feature]
        # )[1]
        p_value = -1
        fisher = stderr_criteria(empirical, reference, fisher_threshold)
        # fisher = distribution_criteria(empirical, reference,fisher_threshold)
        # if target == 'ts-station-service':
        #    print(source,feature,fisher)
        # fisher = fisher_criteria(empirical, reference, side=ALTERNATIVE[feature])
        # if target == 'ts-food-service':
        #     logger.debug(f"{source} {target} {feature} {fisher} "
        #                  f"{np.mean(empirical)} {np.mean(reference)} {np.std(reference)}")
        if fisher:
            useful_features_dict[(source, target)].append(feature)
        # try:
        #     if DEBUG:
        #         import matplotlib.pyplot as plt
        #         from matplotlib.figure import Figure
        #         fig = Figure(figsize=(4, 3))
        #         # x = np.sort(np.concatenate([empirical, reference]))
        #         # print('DEBUG:')
        #         # print(empirical,reference)
        #         sns.distplot(empirical, label='Empirical')
        #         sns.distplot(reference, label='Reference')
        #         plt.xlabel(feature)
        #         plt.ylabel('PDF')
        #         plt.legend()
        #         plt.title(f"{source}->{target}, ks={p_value:.2f}, fisher={fisher:.2f}")
        #         plt.savefig(
        #             plot_dir / f"{input_file.name.split('.')[0]}_{source}_{target}_{feature}.pdf",
        #             bbox_inches='tight', pad_inches=0
        #         )
        # except:
        #     pass
            # logger.debug(f"{input_file.name} {source} {target} {feature} {fisher}")
            # useful_features_dict[(source, target)].append(feature)
    # logger.debug(f"{input_file.name} {dict(useful_features_dict)}")
    with open(output_file, 'w+') as f:
        print(dict(useful_features_dict), file=f)


if __name__ == '__main__':
    input_file = r'E:\AIOPs\TraceRCA-main\A\uninjection\admin-order_abort_1011_data.pkl'
    history = r'E:\AIOPs\TraceRCA-main\A\uninjection\pkl_3_data.pkl'
    output_file = r'E:\AIOPs\TraceRCA-main\A\uninjection\useful_feature_2'
    fisher_threshold = 1
    selecting_feature_main(input_file = input_file,output_file = output_file,history = history,fisher_threshold = fisher_threshold)



