"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-21
"""
from data_utils.metrics import calc_metrics


def eval_model(model, data, metric_meta, use_cuda=True, with_label=True, label_mapper=None):
    """
    :param model:
    :param data:
    :param metric_meta:
    :param use_cuda:
    :param with_label:
    :param label_mapper:
    :return:
    """
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        # 生文本解码
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
    if with_label:
        # 用于dev和test测试用
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    else:
        raise ValueError
    return metrics, predictions, scores, golds, ids


def eval_joint_model(model, data, ner_metric_meta, cls_metric_meta, use_cuda=True, with_label=True,
                     ner_label_mapper=None, cls_label_mapper=None):
    """
    :param model:
    :param data:
    :param ner_metric_meta:
    :param cls_metric_meta:
    :param use_cuda:
    :param with_label:
    :param ner_label_mapper:
    :param cls_label_mapper:
    :return:
    """
    data.reset()
    if use_cuda:
        model.cuda()
    ner_predictions, cls_predictions = [], []
    ner_golds, cls_golds = [], []
    ner_scores, cls_scores = [], []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        # 生文本解码
        score, pred, gold = model.predict(batch_meta, batch_data)
        ner_predictions.extend(pred[0])
        cls_predictions.extend(pred[1])
        ner_golds.extend(gold[0])
        cls_golds.extend(gold[1])
        ner_scores.extend(score[0])
        cls_scores.extend(score[1])
        ids.extend(batch_meta['uids'])
    if with_label:
        # 用于dev和test测试用
        ner_metrics = calc_metrics(ner_metric_meta, ner_golds, ner_predictions, ner_scores, ner_label_mapper)
        cls_metrics = calc_metrics(cls_metric_meta, cls_golds, cls_predictions, cls_scores, cls_label_mapper)
    else:
        raise ValueError
    return ner_metrics, ner_predictions, ner_scores, ner_golds, cls_metrics, cls_predictions, cls_scores, cls_golds, ids


def eval_joint_model_three(model, data, ner_metric_meta, bcls_metric_meta, mcls_metric_meta, use_cuda=True,
                           with_label=True, ner_label_mapper=None, bcls_label_mapper=None, mcls_label_mapper=None):
    """
    :param model:
    :param data:
    :param ner_metric_meta:
    :param bcls_metric_meta:
    :param mcls_metric_meta:
    :param use_cuda:
    :param with_label:
    :param ner_label_mapper:
    :param bcls_label_mapper:
    :param mcls_label_mapper:
    :return:
    """
    data.reset()
    if use_cuda:
        model.cuda()
    ner_predictions, bcls_predictions, mcls_predictions = [], [], []
    ner_golds, bcls_golds, mcls_golds = [], [], []
    ner_scores, bcls_scores, mcls_scores = [], [], []
    ids = []
    for batch_meta, batch_data in data:
        # 生文本解码
        score, pred, gold = model.predict(batch_meta, batch_data)
        ner_predictions.extend(pred[0])
        bcls_predictions.extend(pred[1])
        mcls_predictions.extend(pred[2])
        ner_golds.extend(gold[0])
        bcls_golds.extend(gold[1])
        mcls_golds.extend(gold[2])
        ner_scores.extend(score[0])
        bcls_scores.extend(score[1])
        mcls_scores.extend(score[2])
        ids.extend(batch_meta['uids'])
    if with_label:
        # 用于dev和test测试用
        ner_metrics = calc_metrics(ner_metric_meta, ner_golds, ner_predictions, ner_scores, ner_label_mapper)
        bcls_metrics = calc_metrics(bcls_metric_meta, bcls_golds, bcls_predictions, bcls_scores, bcls_label_mapper)
        mcls_metrics = calc_metrics(mcls_metric_meta, mcls_golds, mcls_predictions, mcls_scores, mcls_label_mapper)
    else:
        raise ValueError
    return ner_metrics, ner_predictions, ner_scores, ner_golds, bcls_metrics, bcls_predictions, bcls_scores, \
           bcls_golds, mcls_metrics, mcls_predictions, mcls_scores, \
           mcls_golds, ids


def eval_joint_model_all(model, data, ner_metric_meta, mt_metric_meta, bcls_metric_meta, mcls_metric_meta,
                         use_cuda, with_label, ner_label_mapper, mt_label_mapper, bcls_label_mapper, mcls_label_mapper):
    """
    :param model:
    :param data:
    :param ner_metric_meta:
    :param mt_metric_meta:
    :param bcls_metric_meta:
    :param mcls_metric_meta:
    :param use_cuda:
    :param with_label:
    :param ner_label_mapper:
    :param mt_label_mapper:
    :param bcls_label_mapper:
    :param mcls_label_mapper:
    :return:
    """
    data.reset()
    if use_cuda:
        model.cuda()
    ner_predictions, mt_predictions, bcls_predictions, mcls_predictions = [], [], [], []
    ner_golds, mt_golds, bcls_golds, mcls_golds = [], [], [], []
    ner_scores, mt_scores, bcls_scores, mcls_scores = [], [], [], []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        # 生文本解码
        score, pred, gold = model.predict(batch_meta, batch_data)
        ner_predictions.extend(pred[0])
        mt_predictions.extend(pred[1])
        bcls_predictions.extend(pred[2])
        mcls_predictions.extend(pred[3])
        ner_golds.extend(gold[0])
        mt_golds.extend(gold[1])
        bcls_golds.extend(gold[2])
        mcls_golds.extend(gold[3])
        ner_scores.extend(score[0])
        mt_scores.extend(score[1])
        bcls_scores.extend(score[2])
        mcls_scores.extend(score[3])
        ids.extend(batch_meta['uids'])
    if with_label:
        # 用于dev和test测试用
        ner_metrics = calc_metrics(ner_metric_meta, ner_golds, ner_predictions, ner_scores, ner_label_mapper)
        mt_metrics = calc_metrics(mt_metric_meta, mt_golds, mt_predictions, mt_scores, mt_label_mapper)
        bcls_metrics = calc_metrics(bcls_metric_meta, bcls_golds, bcls_predictions, bcls_scores, bcls_label_mapper)
        mcls_metrics = calc_metrics(mcls_metric_meta, mcls_golds, mcls_predictions, mcls_scores, mcls_label_mapper)
    else:
        raise ValueError
    return ner_metrics, ner_predictions, ner_scores, ner_golds, mt_metrics, mt_predictions, mt_scores, mt_golds, \
           bcls_metrics, bcls_predictions, bcls_scores, bcls_golds, mcls_metrics, mcls_predictions, mcls_scores,\
           mcls_golds, ids
