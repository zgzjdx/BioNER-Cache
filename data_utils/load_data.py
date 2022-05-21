"""
author:yqtong@stu.xmu.edu.cn
date:2020-12-24
"""
import json
import numpy as np
from data_utils.task_def import TaskType, DataFormat, EncoderModelType


def load_data(file_path, data_format, task_type, label_dict):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_dict: map string label to numbers, only valid for Cls
     and Ranking task (for ranking task, better label should have large numbers)
    :return:
    """
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    for line in open(file_path, encoding='utf-8'):
        fields = line.strip().split('\t')
        if data_format == DataFormat.PremiseOnly:
            # uid, label, premise
            assert len(fields) == 3
            row = {'uid': fields[0], 'label': fields[1], 'premise': fields[2]}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            # uid, label, premise, hypothesis
            assert len(fields) == 4
            row = {'uid': fields[0], 'label': fields[1], 'premise': fields[2], 'hypothesis': fields[3]}
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {'uid': fields[0], 'ruid': fields[1].split(','), 'label': fields[2], 'premise': fields[3],
                   'hypothesis': fields[4:]}
        elif data_format == DataFormat.Sequence:
            assert len(fields) == 3
            row = {'uid': fields[0], 'label': json.loads(fields[1]), 'premise': json.loads(fields[2])}
        elif data_format == DataFormat.MRC:
            assert len(fields) == 4
            row = {'uid': fields[0], 'label': fields[1], 'premise': fields[2], 'hypothesis': fields[3]}
        elif data_format == DataFormat.OneMRCAndOneSequence:
            assert len(fields) == 8
            # premise 正文 hypothesis query label ner_label
            row = {'uid': fields[0], 'domain_label': fields[1], 'premise': fields[2], 'hypothesis': fields[3], 'label': fields[4],
                   'start_position': fields[5], 'end_position': fields[6], 'span_position': fields[7]}
        else:
            raise ValueError(data_format)

        # 将标签映射为one-hot
        if task_type == TaskType.Classification:
            if label_dict is not None:
                row['label'] = label_dict[row['label']]
            else:
                row['label'] = int(row['label'])
        elif task_type == TaskType.Ranking:
            labels = row['label'].split(',')
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            # argmax返回最大值的索引
            row['label'] = int(np.argmax(labels))
            row['orign_label'] = labels
        elif task_type == TaskType.ReadingComprehension:
            pass
        elif task_type == TaskType.SequenceLabeling:
            if label_dict is not None:
                row['label'] = [label_dict[l] for l in row['label']]
            else:
                row['label'] = [int(l) for l in row['label']]
        elif task_type == TaskType.Joint_mrc_ner:
            # type list
            label_list = eval(row['label'])
            if label_dict is not None:
                row['ner_label'] = [label_dict[l] for l in label_list]
            else:
                raise ValueError
        assert row['label'] is not None

        rows.append(row)
    return rows


def load_joint_data_two(file_path, data_format, task_type, label_dict_a, label_dict_b):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_dict_a:
    :param label_dict_b:
    :return:
    """
    rows = []
    for line in open(file_path, encoding='utf-8'):
        fields = line.strip().split('\t')
        if data_format == DataFormat.OnePremiseAndOneSequence:
            assert len(fields) == 4
            row = {'uid': fields[0], 'cls_label': fields[1], 'ner_label': json.loads(fields[2]),
                   'premise': json.loads(fields[3])}
        elif data_format == DataFormat.OneSequenceAndOneSequence:
            assert len(fields) == 4
            row = {'uid': fields[0], 'ner_label': json.loads(fields[1]), 'mt_label': json.loads(fields[2]),
                   'premise': json.loads(fields[3])}
        elif data_format == DataFormat.TwoPremise:
            assert len(fields) == 4
            row = {'uid': fields[0], 'bcls_label': fields[1], 'mcls_label': fields[2],
                   'premise': fields[3]}
        else:
            raise ValueError

        if task_type == TaskType.Joint:
            if label_dict_a is not None:
                row['cls_label'] = label_dict_a[row['cls_label']]
            else:
                row['cls_label'] = int(row['cls_label'])

            if label_dict_b is not None:
                row['ner_label'] = [label_dict_b[l] for l in row['ner_label']]
            else:
                row['ner_label'] = [int(l) for l in row['ner_label']]
        elif task_type == TaskType.Joint_mt:
            if label_dict_a is not None:
                row['mt_label'] = [label_dict_a[l] for l in row['mt_label']]
            else:
                row['mt_label'] = [int(l) for l in row['mt_label']]
            if label_dict_b is not None:
                row['ner_label'] = [label_dict_b[l] for l in row['ner_label']]
            else:
                row['ner_label'] = [int(l) for l in row['ner_label']]
        elif task_type == TaskType.Joint_bCLS_mCLS:
            if label_dict_a is not None:
                row['bcls_label'] = label_dict_a[row['bcls_label']]
            else:
                row['bcls_label'] = int(row['bcls_label'])
            if label_dict_b is not None:
                row['mcls_label'] = label_dict_b[row['mcls_label']]
            else:
                row['mcls_label'] = int(row['mcls_label'])
        else:
            raise ValueError

        rows.append(row)
    return rows


def load_joint_data_three(file_path, data_format, task_type, label_dict_a, label_dict_b, label_dict_c):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_dict_a:
    :param label_dict_b:
    :param label_dict_c:
    :return:
    """
    rows = []
    for line in open(file_path, encoding='utf-8'):
        fields = line.strip().split('\t')
        if data_format == DataFormat.OnePremiseAndTwoSequence:
            assert len(fields) == 5
            row = {'uid': fields[0], 'cls_label': fields[1], 'ner_label': json.loads(fields[2]),
                   'mt_label': json.loads(fields[3]), 'premise': json.loads(fields[4])}
        elif data_format == DataFormat.TwoPremiseAndOneSequence:
            assert len(fields) == 5
            row = {'uid': fields[0], 'bcls_label': fields[1], 'mcls_label': fields[2],
                   'ner_label': json.loads(fields[3]), 'premise': json.loads(fields[4])}
        else:
            raise ValueError

        if task_type == TaskType.Joint_mt_three:
            # cls, mt, ner
            if label_dict_a is not None:
                row['cls_label'] = label_dict_a[row['cls_label']]
            else:
                row['cls_label'] = int(row['cls_label'])
            if label_dict_b is not None:
                row['mt_label'] = [label_dict_b[l] for l in row['mt_label']]
            else:
                row['mt_label'] = [int(l) for l in row['mt_label']]
            if label_dict_c is not None:
                row['ner_label'] = [label_dict_c[l] for l in row['ner_label']]
            else:
                row['ner_label'] = [int(l) for l in row['ner_label']]
        elif task_type == TaskType.Joint_cls_three:
            # bcls, mcls, ner
            if label_dict_a is not None:
                row['bcls_label'] = label_dict_a[row['bcls_label']]
            else:
                row['bcls_label'] = int(row['bcls_label'])
            if label_dict_b is not None:
                row['mcls_label'] = label_dict_b[row['mcls_label']]
            else:
                row['mcls_label'] = int(row['mcls_label'])
            if label_dict_c is not None:
                row['ner_label'] = [label_dict_c[l] for l in row['ner_label']]
            else:
                row['ner_label'] = [int(l) for l in row['ner_label']]
        else:
            raise ValueError

        rows.append(row)
    return rows


def load_joint_data_all_cls(file_path, data_format, task_type, label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_mapper_bcls:
    :param label_mapper_mcls:
    :param label_mapper_ner:
    :param label_mapper_mt:
    :return:
    """
    rows = []
    for line in open(file_path, encoding='utf-8'):
        fields = line.strip().split('\t')
        if data_format == DataFormat.TwoPremiseAndTwoSequence:
            assert len(fields) == 6
            row = {'uid': fields[0], 'bcls_label': fields[1], 'mcls_label': fields[2], 'ner_label': json.loads(fields[3]),
                   'mt_label': json.loads(fields[4]), 'premise': json.loads(fields[5])}
        else:
            raise ValueError

        if task_type == TaskType.Joint_all_cls:
            if label_mapper_bcls is not None:
                row['bcls_label'] = label_mapper_bcls[row['bcls_label']]
            else:
                row['bcls_label'] = int(row['bcls_label'])
            if label_mapper_mcls is not None:
                row['mcls_label'] = label_mapper_mcls[row['mcls_label']]
            else:
                row['mcls_label'] = int(row['mcls_label'])
            if label_mapper_mt is not None:
                row['mt_label'] = [label_mapper_mt[l] for l in row['mt_label']]
            else:
                row['mt_label'] = [int(l) for l in row['mt_label']]
            if label_mapper_ner is not None:
                row['ner_label'] = [label_mapper_ner[l] for l in row['ner_label']]
            else:
                row['ner_label'] = [int(l) for l in row['ner_label']]
        else:
            raise ValueError

        rows.append(row)
    return rows


def load_joint_data_all(file_path, data_format, task_type, label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_mapper_bcls:
    :param label_mapper_mcls:
    :param label_mapper_ner:
    :param label_mapper_mt:
    :return:
    """
    rows = []
    for line in open(file_path, encoding='utf-8'):
        fields = line.strip().split('\t')
        if data_format == DataFormat.TwoPremiseAndTwoSequenceAndMRC:
            assert len(fields) == 11
            # id type_label 正文 query bCLS_label mCLS_label mtCLS_label, ner_label, start_position, end_position, span_position
            row = {'uid': fields[0], 'domain_label': fields[1], 'premise': fields[2], 'hypothesis': fields[3],
                   'bCLS_label': fields[4], 'mCLS_label': fields[5], 'mtCLS_label': fields[6], 'ner_label': fields[7],
                   'start_position': fields[8], 'end_position': fields[9], 'span_position': fields[10]}
        else:
            raise ValueError

        if task_type == TaskType.Joint_all:
            if label_mapper_bcls is not None:
                row['bCLS_label'] = label_mapper_bcls[row['bCLS_label']]
            else:
                row['bCLS_label'] = int(row['bCLS_label'])
            if label_mapper_mcls is not None:
                row['mCLS_label'] = label_mapper_mcls[row['mCLS_label']]
            else:
                row['mCLS_label'] = int(row['mCLS_label'])
            if label_mapper_mt is not None:
                row['mtCLS_label'] = [label_mapper_mt[l] for l in eval(row['mtCLS_label'])]
            else:
                row['mtCLS_label'] = [int(l) for l in eval(row['mtCLS_label'])]
            if label_mapper_ner is not None:
                row['ner_label'] = [label_mapper_ner[l] for l in eval(row['ner_label'])]
            else:
                row['ner_label'] = [int(l) for l in eval(row['ner_label'])]
        else:
            raise ValueError

        rows.append(row)
    return rows