"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-15
"""
from enum import Enum
from collections import OrderedDict
from sklearn.metrics import matthews_corrcoef
# 马修斯相关系数
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from data_utils.mrc_eval import evaluate_func
from data_utils.sequence_labeling import classification_report
import numpy as np
import torch
from data_utils.pmetrics import ner_report_conlleval, blue_classification_report
import os


def compute_acc(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    return 100.0 * accuracy_score(y_true, y_pred)


def compute_f1(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    return np.array(100.0 * f1_score(y_true, y_pred)).item(), np.array(100.0 * precision_score(y_true, y_pred)).item(), \
           np.array(100.0 * recall_score(y_true, y_pred)).item()


def compute_mcc(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    return matthews_corrcoef(y_true, y_pred)


def compute_pearson(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    pcof = pearsonr(y_true, y_pred)[0]
    return 100.0 * pcof


def compute_spearman(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    scof = spearmanr(y_true, y_pred)[0]
    return 100.0 * scof


def compute_auc(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    auc = roc_auc_score(y_true, y_pred)
    return 100.0 * auc


def compute_micro_f1(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    report = blue_classification_report(y_true, y_pred)
    # np.ndarray.item() copy an element of an array to a standard Python
    return report.micro_row.f1.item()


def compute_micro_f1_subindex(y_pred, y_true, subindex):
    """
    :param y_pred:
    :param y_true:
    :param subindex:
    :return:
    """
    pass


def compute_macro_f1_subindex(y_pred, y_true, subindex):
    """
    :param y_pred:
    :param y_true:
    :param subindex:
    :return:
    """
    pass


def compute_seq_f1(predicts, labels, label_mapper):
    """
    :param predicts:
    :param labels:
    :param label_mapper:
    :return:
    """
    y_true, y_pred = [], []
    def trim(predict, label):
        temp_1, temp_2 = [], []
        for index, m in enumerate(predict):
            # 忽略CLS
            if index == 0:
                continue
            if label_mapper[label[index]] != 'X':
                # 忽略因subword操作产生的X标签
                temp_1.append(label_mapper[label[index]])
                temp_2.append(label_mapper[m])
        # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        temp_1.pop()
        temp_2.pop()
        y_true.append(temp_1)
        y_pred.append(temp_2)
    for predict, label in zip(predicts, labels):
        trim(predict, label)
    # print(y_true)
    # print(y_pred)
    # report = classification_report(y_true, y_pred, digits=4)
    report = ner_report_conlleval(y_true, y_pred)
    return report.micro_row.f1.item(), report.micro_row.precision.item(), report.micro_row.recall.item()


def compute_emf1(predicts, labels):
    return evaluate_func(labels, predicts)


def compute_spanf1(predictions, golds):
    """
    :param predictions: match_preds
    :param golds: match_labels
    :return:
    """
    assert len(predictions) == len(golds)
    outputs = []
    for predict, gold in zip(predictions, golds):
        tp = (gold & predict).long().sum()
        fp = (~gold & predict).long().sum()
        fn = (gold & ~predict).long().sum()
        outputs.append(torch.stack([tp, fp, fn]))
    all_counts = torch.stack([x for x in outputs]).sum(0)
    span_tp, span_fp, span_fn = all_counts
    span_recall = span_tp / (span_tp + span_fn + 1e-10)
    span_precision = span_tp / (span_tp + span_fp + 1e-10)
    span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
    # print('debug use')
    return span_f1.item(), span_precision.item(), span_recall.item()


def compute_seq_cls_f1(predicts, labels, label_mapper):
    """
    :param predicts:
    :param labels:
    :param label_mapper:
    :return:
    """
    scores = []
    def trim(predict, label):
        temp_1, temp_2 = [], []
        for index, m in enumerate(predict):
            # 忽略CLS
            if index == 0:
                continue
            if label_mapper[label[index]] != 'X':
                # 忽略因subword操作产生的X标签
                temp_1.append(label[index]) # y_true
                temp_2.append(m) # y_pred
        # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
        # 忽略SEP
        temp_1.pop()
        temp_2.pop()
        report = blue_classification_report(temp_1, temp_2)
        scores.append(report.micro_row.f1.item())

    for predict, label in zip(predicts, labels):
        trim(predict, label)
    return sum(scores) / len(scores)


class Metric(Enum):
    ACC = 0
    F1 = 1
    MCC = 2
    Pearson = 3
    Spearman = 4
    AUC = 5
    SeqEval = 7
    MicroF1 = 8
    MicroF1WithoutLastOne = 9
    MacroF1WithoutLastOne = 10
    EmF1 = 11
    SpanF1 = 12
    SeqClsEval = 13


METRIC_FUNC = {
    Metric.ACC: compute_acc,
    Metric.F1: compute_f1,
    Metric.MCC: compute_mcc,
    Metric.Pearson: compute_pearson,
    Metric.Spearman: compute_spearman,
    Metric.AUC: compute_auc,
    Metric.SeqEval: compute_seq_f1,
    Metric.MicroF1: compute_micro_f1,
    Metric.MicroF1WithoutLastOne: compute_micro_f1_subindex,
    Metric.MacroF1WithoutLastOne: compute_macro_f1_subindex,
    Metric.EmF1: compute_emf1,
    Metric.SpanF1: compute_spanf1,
    Metric.SeqClsEval: compute_seq_cls_f1
}


def calc_metrics(metrics_meta, golds, predictions, scores, label_mapper=None):
    """
    :param metrics_meta:
    :param golds:
    :param predictions:
    :param scores:
    :param label_mapper:
    :return:
    """
    metrics = OrderedDict()
    for mm in metrics_meta:
        # print(mm)
        metric_name = mm.name
        metric_func = METRIC_FUNC[mm]
        if mm in (Metric.ACC, Metric.MCC, Metric.MicroF1):
            metric = metric_func(predictions, golds)
        elif mm == Metric.F1:
            metric, precision, recall = metric_func(predictions, golds)
            metrics['precision'] = precision
            metrics['recall'] = recall
        elif mm == Metric.SeqClsEval:
            metric = metric_func(predictions, golds, label_mapper)
        elif mm == Metric.SeqEval:
            metric, precision, recall = metric_func(predictions, golds, label_mapper)
            metrics['precision'] = precision
            metrics['recall'] = recall
        elif mm == Metric.EmF1:
            metric = metric_func(predictions, golds)
        elif mm == Metric.SpanF1:
            # predictions = match_preds
            # golds = match_labels
            metric, precision, recall = metric_func(predictions, golds)
            metrics['precision'] = precision
            metrics['recall'] = recall
        elif mm == Metric.MicroF1WithoutLastOne:
            metric = metric_func(predictions, golds, subindex=list(range(len(label_mapper) - 1)))
        elif mm == Metric.MacroF1WithoutLastOne:
            metric = metric_func(predictions, golds, subindex=list(range(len(label_mapper) - 1)))
        else:
            if mm == Metric.AUC:
                assert len(scores) == 2 * len(golds)
                scores = scores[1::2]
                # [start:end:step]
            metric = metric_func(predictions, golds, label_mapper)
        metrics[metric_name] = metric
    return metrics


if __name__ == '__main__':
    # y_true = [1, 0, 1, 1, 1, 2, 2, 2, 2, 3, 1, 0]
    # y_pred = [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2]
    y_pred = [1, 2, 1, 2, 2, 2, 2, 1, 2]
    result1 = compute_micro_f1(y_pred, y_true)
    print(type(result1))
    print(result1)
    result2, result21, result22 = compute_f1(y_pred, y_true)
    print(type(result2))
    print(result2)
    from sklearn.metrics import precision_score, recall_score, f1_score
    import numpy as np
    result4 = f1_score(y_pred, y_true, average='micro')
    print(type(result4))
    print(result4)
    result5 = precision_score(y_pred, y_true, average='micro')
    print(type(result5))
    print(result5)
    result6 = recall_score(y_pred, y_true, average='micro')
    print(type(result6))
    print(result6)
    result7 = f1_score(y_pred, y_true)
    print(np.array(result7))
    print(type(result7))
    result8 = precision_score(y_pred, y_true)
    print(np.array(result8))
    result9 = recall_score(y_pred, y_true)
    print(np.array(result9))




