import pickle
import time
from pathlib import Path

import click
import numpy as np
from sklearn.ensemble import IsolationForest
from loguru import logger
# from trainticket_config import FEATURE_NAMES
FEATURE_NAMES = ['latency','http_status','cpu_use','mem_use_percent','mem_use_amount','file_write_rate','file_read_rate',
                 'net_send_rate','net_receive_rate']
# from diskcache import Cache

DEBUG = True

threshold = 1.0


def anomaly_detection_isolation_forest(df, result_column, history, cache):
    """
    This function has not been used

    """
    indices = np.unique(df.index.values) #每个调用
    for source, target in indices:
        empirical = df.loc[(source, target), FEATURE_NAMES].values
        # reference = history.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        if token not in cache:
            df.loc[(source, target), result_column] = 0
            continue
        model = cache[token]
        predict = model.predict(empirical)
        df.loc[(source, target), result_column] = predict
    return df


def anomaly_detection_3sigma_without_useful_features(df, result_column, history, cache):
    """This function has not been used
    """
    indices = np.unique(df.index.values)
    useful_feature = {key: FEATURE_NAMES for key in indices}
    return anomaly_detection_3sigma(df, result_column, None, useful_feature, cache=cache)


def anomaly_detection_3sigma(df, result_column, history, useful_feature, cache):
    """Multi-Metric Invocation Anomaly Detection

    :param df: The data after the fault happens
    :param result_column: The column name of the result of the prediction method we used
    :param history: Not used
    :param useful_feature: the useful features of each invocation (dict)
    :param cache: Not used
    :return: the data with the anomally invocations detection
    """
    indices = np.unique(df.index.values)
    for source, target in indices:
        if (source, target) not in useful_feature:  # all features are not useful
            df.loc[(source, target), result_column] = 0
            continue
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values #取出有用的特征
        mean, std = [], []
        for idx, feature in enumerate(features):
            token = f"reference-{source}-{target}-{feature}-mean-variance"
            # if token in cache:
            #     mean.append(cache[token]['mean'])
            #     std.append(cache[token]['std'])
            # else:
            mean.append(np.mean(empirical,axis=0)[idx])
            std.append(np.maximum(np.std(empirical,axis=0)[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, feature in enumerate(features):
            predict[:, idx] = np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx] #判断该feature是否异常
        predict = np.max(predict, axis=1) #选取该调用的各feature严重程度的最大值,只要有一个feature是异常的，该调用就是异常的

        df.loc[(source, target), result_column] = predict
    return df


# @click.command('invocation anomaly detection')
# @click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
# @click.option('-o', '--output', 'output_file', default='.', type=str)
# @click.option('-h', '--history', default='historical_data.pkl', type=str)
# @click.option('-u', '--useful-feature', "useful_feature", default='.', type=str)
# @click.option('-c', '--cache', 'cache_file', default='.', type=str)
# @click.option('-t', '--threshold', 'main_threshold', default=1, type=float)





def invo_anomaly_detection_main(input_file, output_file, history, useful_feature, cache_file, main_threshold):
    """The main function to select the useful features

    :param input_file: The data after the fault happens
    :param output_file: The data with the anomaly invocations detection
    :param history: Not used
    :param useful_feature: The useful features of each invocation (dict)
    :param cache_file: Not used
    :param main_threshold: a given threshold to compare the anomaly severity
    :return: The data with the anomaly invocations detection
    """
    global threshold
    threshold = main_threshold

    history = None
    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))
    # logger.debug(f"useful features: {useful_feature}")

    # with open(cache_file, 'rb+') as f:
    #     cache = pickle.load(f)
    cache = None

    input_file = Path(input_file)

    with open(input_file, 'rb') as f:
        df = pickle.load(f)
    df = df.set_index(keys=['source', 'target'], drop=False).sort_index()
    # history = history.set_index(keys=['source', 'target'], drop=False).sort_index()
    tic = time.time()
    # df = anomaly_detection_3sigma(df, 'Ours-predict', None, useful_feature, cache=cache)
    df = anomaly_detection_3sigma(df, 'Ours-predict', None, useful_feature,cache=cache)
    toc = time.time()
    print("algo:", "ours", "time:", toc - tic, 'invos:', len(df))

    # df = anomaly_detection_3sigma_without_useful_features(df, 'NoSelection-predict', None, cache=cache)
    #
    # # tic = time.time()
    # df = anomaly_detection_isolation_forest(df, 'IF-predict', None, cache=cache)
    # # toc = time.time()
    # # print("algo:", "IF", "time:", toc - tic, 'invos:', len(df))

    df['predict'] = df['Ours-predict']
    with open(output_file, 'wb+') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    input_file = r'E:\AIOPs\TraceRCA-main\A\uninjection\admin-order_abort_1011_data.pkl'
    history = r'E:\AIOPs\TraceRCA-main\A\uninjection\pkl_3_data.pkl'
    output_file = r'E:\AIOPs\TraceRCA-main\A\uninjection\invo_anomaly_detection_2.pkl'
    useful_feature = r'E:\AIOPs\TraceRCA-main\A\uninjection\useful_feature_2'
    main_threshold = 1


    invo_anomaly_detection_main(input_file = input_file,output_file = output_file,history = history,useful_feature = useful_feature,
                                main_threshold = main_threshold,cache_file = None)

