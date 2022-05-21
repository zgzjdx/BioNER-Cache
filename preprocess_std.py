"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-16
"""
import json
import os
from data_utils.load_data import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from data_utils.logger_wrapper import create_logger
from multi_exp_def import MultiTaskDefs
from mt_dnn.batcher import BatchGen
from data_utils.buid_data import *


def build_data(data, dump_path, tokenizer, mrc_tokenizer, data_format=DataFormat.PremiseOnly, max_seq_length=MAX_SEQ_LEN,
               encoderModelType=EncoderModelType.BERT, task_type=None, lab_dict=None):
    """
    :param data:
    :param dump_path:
    :param tokenizer:
    :param mrc_tokenizer:
    :param data_format:
    :param max_seq_length:
    :param encoderModelType:
    :param task_type:
    :param lab_dict:
    :return:
    """
    if data_format == DataFormat.PremiseOnly:
        assert task_type == TaskType.Classification
        build_data_premise_only(data, dump_path, max_seq_length, tokenizer)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(data, dump_path, max_seq_length, tokenizer)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(data, dump_path, max_seq_length, tokenizer)
    elif data_format == DataFormat.Sequence:
        assert task_type == TaskType.SequenceLabeling
        build_data_sequence(data, dump_path, max_seq_length, tokenizer, lab_dict)
    elif data_format == DataFormat.MRC:
        assert task_type == TaskType.ReadingComprehension
        build_data_mrc(data, dump_path, max_seq_length, tokenizer, encoderModelType)
    elif data_format == DataFormat.OneMRCAndOneSequence:
        assert task_type == TaskType.Joint_mrc_ner
        build_data_mrc_ner(data, dump_path, max_seq_length, tokenizer, mrc_tokenizer, encoderModelType, lab_dict)
    else:
        raise ValueError(data_format)


def build_joint_data_two(data, dump_path, tokenizer, data_format, max_seq_length, encoderModelType, task_type,
                     lab_dict_a, lab_dict_b):
    """
    :param data:
    :param dump_path:
    :param tokenizer:
    :param data_format:
    :param max_seq_length:
    :param encoderModelType:
    :param task_type:
    :param lab_dict_a:
    :param lab_dict_b:
    :return:
    """
    if data_format == DataFormat.OnePremiseAndOneSequence:
        assert task_type == TaskType.Joint
        build_data_joint_two(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_b)
    elif data_format == DataFormat.OneSequenceAndOneSequence:
        assert task_type == TaskType.Joint_mt
        build_data_joint_mt(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_a, lab_dict_b)
    elif data_format == DataFormat.TwoPremise:
        assert task_type == TaskType.Joint_bCLS_mCLS
        build_data_joint_bcls_mcls(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_a, lab_dict_b)
    else:
        raise ValueError


def build_joint_data_three(data, dump_path, tokenizer, data_format, max_seq_length, encoderModelType, task_type,
                     lab_dict_a, lab_dict_b, lab_dict_c):
    """
    :param data:
    :param dump_path:
    :param tokenizer:
    :param data_format:
    :param max_seq_length:
    :param encoderModelType:
    :param task_type:
    :param lab_dict_a:
    :param lab_dict_b:
    :param lab_dict_c:
    :return:
    """
    if data_format == DataFormat.OnePremiseAndTwoSequence:
        assert task_type == TaskType.Joint_mt_three
        build_data_joint_three_mt(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_b, lab_dict_c)
    elif data_format == DataFormat.TwoPremiseAndOneSequence:
        assert task_type == TaskType.Joint_cls_three
        build_data_joint_three_cls(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_c)
    else:
        raise ValueError


def build_joint_data_all_cls(data, dump_path, tokenizer, data_format, max_seq_length, encoderModelType, task_type,
                     label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt):
    """
    :param data:
    :param dump_path:
    :param tokenizer:
    :param data_format:
    :param max_seq_length:
    :param encoderModelType:
    :param task_type:
    :param label_mapper_bcls:
    :param label_mapper_mcls:
    :param label_mapper_ner:
    :param label_mapper_mt:
    :return:
    """
    if data_format == DataFormat.TwoPremiseAndTwoSequence:
        assert task_type == TaskType.Joint_all_cls
        build_data_joint_all_cls(data, dump_path, max_seq_length, tokenizer, encoderModelType, label_mapper_ner, label_mapper_mt)
    else:
        raise ValueError


def build_joint_data_all(data, dump_path, tokenizer, mrc_tokenizer, data_format, max_seq_len, encoderModelType, task_type,
                                     label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt):
    """
    :param data:
    :param dump_path:
    :param tokenizer:
    :param mrc_tokenizer:
    :param data_format:
    :param max_seq_len:
    :param encoderModelType:
    :param task_type:
    :param label_mapper_bcls:
    :param label_mapper_mcls:
    :param label_mapper_ner:
    :param label_mapper_mt:
    :return:
    """
    if data_format == DataFormat.TwoPremiseAndTwoSequenceAndMRC:
        assert task_type == TaskType.Joint_all
        build_data_joint_all(data, dump_path, max_seq_len, tokenizer, mrc_tokenizer,
                             encoderModelType, label_mapper_ner, label_mapper_mt)
    else:
        raise ValueError


def data_preprocess_std(data_path, bert_path, task_name):
    """
    :param data_path:
    :param bert_path:
    :param task_name:
    :return:
    """
    root = data_path
    assert os.path.exists(root)
    log_file = os.path.join(root, f'preprocess_std_{MAX_SEQ_LEN}.log')
    logger = create_logger(__name__, to_disk=True, log_file=log_file)
    # 默认是区分大小写的, do_lower_case = False
    is_uncased = False
    if 'uncased' in bert_path:
        # 不区分大小写, do_lower_case = True
        is_uncased = True

    mt_dnn_suffix = 'bert'
    encoder_model = EncoderModelType.BERT
    # print(encoder_model.value)
    vocab_path = os.path.join(bert_path, 'vocab.txt')
    tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=is_uncased)

    if is_uncased:
        # 不做大小写区分的, 即统一小写化处理
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_file_path = 'multi_task_def.yml'
    task_defs = MultiTaskDefs(task_file_path)

    if task_name == 'all':
        tasks = task_defs.task
    else:
        tasks = task_name.split(',')

    for task in tasks:
        logger.info('Current task %s', task)
        data_format = task_defs.data_format_map[task]
        task_type = task_defs.task_type_map[task]
        # todo 简化代码, 下面的应该用task_type来区分
        if 'Joint-two' in task or 'Joint-bCLS-mtCLS' in task or 'Joint-mCLS-mtCLS' in task:
            label_mapper_a = task_defs.cls_label_mapper_map[task]
            label_mapper_b = task_defs.ner_label_mapper_map[task]
        elif 'Joint-mt' in task:
            label_mapper_a = task_defs.mt_label_mapper_map[task]
            label_mapper_b = task_defs.ner_label_mapper_map[task]
        elif 'Joint-three-mt' in task:
            label_mapper_a = task_defs.cls_label_mapper_map[task]
            label_mapper_b = task_defs.mt_label_mapper_map[task]
            label_mapper_c = task_defs.ner_label_mapper_map[task]
        elif 'Joint-three-cls' in task:
            label_mapper_a = task_defs.bcls_label_mapper_map[task]
            label_mapper_b = task_defs.mcls_label_mapper_map[task]
            label_mapper_c = task_defs.ner_label_mapper_map[task]
        elif 'Joint-all-cls' in task:
            label_mapper_bcls = task_defs.bcls_label_mapper_map[task]
            label_mapper_mcls = task_defs.mcls_label_mapper_map[task]
            label_mapper_ner = task_defs.ner_label_mapper_map[task]
            label_mapper_mt = task_defs.mt_label_mapper_map[task]
        elif 'Joint-bCLS-mCLS' in task:
            label_mapper_a = task_defs.bcls_label_mapper_map[task]
            label_mapper_b = task_defs.mcls_label_mapper_map[task]
        elif 'MRC-rule' in task or 'MRC-wiki' in task or 'MRC-ours' in task or 'MRC-none' in task:
            label_mapper = task_defs.ner_label_mapper_map[task]
        elif 'Joint-all-rule' in task:
            # bCLS+mCLS+mtCLS+cache+MRC+ner
            label_mapper_bcls = task_defs.bcls_label_mapper_map[task]
            label_mapper_mcls = task_defs.mcls_label_mapper_map[task]
            label_mapper_mt = task_defs.mt_label_mapper_map[task]
            label_mapper_ner = task_defs.ner_label_mapper_map[task]
        else:
            label_mapper = task_defs.label_mapper_map[task]
        split_names = task_defs.split_names_map[task]
        for split_name in split_names:
            dump_path = os.path.join(mt_dnn_root, f'{task}_{split_name}.json')
            if os.path.exists(dump_path):
                logger.warning('Dump path %s exists!' % dump_path)
            if 'Joint-two' in task or 'Joint-mt' in task or 'Joint-bCLS-mCLS' in task or \
                'Joint-bCLS-mtCLS' in task or 'Joint-mCLS-mtCLS' in task:
                rows = load_joint_data_two(os.path.join(root, f'{task}_{split_name}.tsv'), data_format,
                                           task_type, label_mapper_a, label_mapper_b)
                build_joint_data_two(rows, dump_path, tokenizer, data_format, max_seq_length=MAX_SEQ_LEN,
                           encoderModelType=encoder_model, task_type=task_type, lab_dict_a=label_mapper_a, lab_dict_b=label_mapper_b)
            elif 'Joint-three' in task:
                rows = load_joint_data_three(os.path.join(root, f'{task}_{split_name}.tsv'),
                                             data_format, task_type, label_mapper_a, label_mapper_b, label_mapper_c)
                build_joint_data_three(rows, dump_path, tokenizer, data_format, max_seq_length=MAX_SEQ_LEN,
                           encoderModelType=encoder_model, task_type=task_type, lab_dict_a=label_mapper_a, lab_dict_b=label_mapper_b, lab_dict_c=label_mapper_c)
            elif 'Joint-all-cls' in task:
                rows = load_joint_data_all_cls(os.path.join(root, f'{task}_{split_name}.tsv'), data_format, task_type,
                                           label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt)
                build_joint_data_all_cls(rows, dump_path, tokenizer, data_format, MAX_SEQ_LEN, encoder_model, task_type,
                                     label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt)
            elif 'Joint-all-rule' in task:
                rows = load_joint_data_all(os.path.join(root, f'{task}_{split_name}.tsv'), data_format, task_type,
                                                label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt)
                mrc_tokenizer = BertWordPieceTokenizer(vocab=vocab_path, lowercase=is_uncased)
                build_joint_data_all(rows, dump_path, tokenizer, mrc_tokenizer, data_format, MAX_SEQ_LEN, encoder_model,
                                     task_type, label_mapper_bcls, label_mapper_mcls, label_mapper_ner, label_mapper_mt)
            else:
                rows = load_data(os.path.join(root, f'{task}_{split_name}.tsv'), data_format, task_type, label_mapper)
                logger.info('%s: Loaded %s %s samples', task, len(rows), split_name)
                if task_type == TaskType.SequenceLabeling:
                    # todo 滑窗策略
                    rows = split_if_longer(rows, label_mapper, 30)
                if task_type == TaskType.ReadingComprehension or task_type == TaskType.Joint_mrc_ner:
                    mrc_tokenizer = BertWordPieceTokenizer(vocab=vocab_path, lowercase=is_uncased)
                build_data(rows, dump_path, tokenizer, None, data_format, max_seq_length=MAX_SEQ_LEN,
                       encoderModelType=encoder_model, task_type=task_type, lab_dict=label_mapper)
        logger.info('%s: Done', task)

    # Follow with previous works, combine train and dev
    # for task in tasks:
    #     logger.info("Current task %s: combine train and dev" % task)
    #     dump_path = os.path.join(mt_dnn_root, f"{task}_train_dev.json")
    #     if os.path.exists(dump_path):
    #         logger.warning('%s: Now overwrite train_dev.json: %s', task, dump_path)
    #         continue
    #     task_type = task_defs.task_type_map[task]
    #     train_path = os.path.join(mt_dnn_root, f'{task}_train.json')
    #     dev_path = os.path.join(mt_dnn_root, f"{task}_dev.json")
    #     train_rows = BatchGen.load(train_path, maxlen=MAX_SEQ_LEN, task_type=task_type)
    #     dev_rows = BatchGen.load(dev_path, maxlen=MAX_SEQ_LEN, task_type=task_type)
    #
    #     with open(dump_path, 'w', encoding='utf-8') as fb:
    #         for features in train_rows + dev_rows:
    #             fb.write('{}\n'.format(json.dumps(features)))
    #     logger.info('%s: Done', task)


if __name__ == '__main__':
    data_path = './canonical_data'
    bert_path = './pretrained_models/biobert_base_cased'
    # bert_path = './pretrained_models/chinesebert_base_uncased'
    task_name = 'all'
    data_preprocess_std(data_path, bert_path, task_name)
