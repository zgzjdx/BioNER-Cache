"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-16
"""
import csv
import json
import logging

from data_utils.task_def import DataFormat


def load_relation(file, dataformat):
    """"""
    # 一般把关系抽取看做句子级的分类任务
    rows = []
    with open(file, encoding='utf-8') as fa:
        reader = csv.reader(fa, delimiter='\t')
        header = next(reader)
        for index, row in enumerate(reader):
            assert len(row) == 3
            # row[0]=index, row[1]=sentence, row[2]=label
            label = row[-1]
            sample = {'uid': row[0], 'premise': row[1], 'label': label}
            rows.append(sample)
    return rows


def load_cls(file):
    """
    :param file:
    :return:
    """
    # 一般把关系抽取看做句子级的分类任务
    rows = []
    with open(file, encoding='utf-8') as fa:
        line_list = fa.readlines()
        for index, row in enumerate(line_list):
            row = row.strip().split('\t')
            assert len(row) == 2
            # row[0]=label, row[1]=premise
            label = row[0]
            sample = {'uid': str(index), 'premise': row[1], 'label': label}
            rows.append(sample)
    print('rows_length', len(rows))
    return rows


def load_mednli(file):
    """
    :param file:
    :return:
    """
    rows = []
    with open(file, 'r', encoding='utf-8') as fa:
        reader = csv.reader(fa, delimiter='\t')
        header = next(reader)
        for index, row in enumerate(reader):
            assert len(row) == 4
            label = row[0]
            assert label is not None
            sample = {'uid': row[1], 'premise': row[2], 'hypothesis': row[3], 'label': label}
            rows.append(sample)
    return rows


def load_sts(file):
    """
    :param file:
    :return:
    """
    rows = []
    cnt = 0
    with open(file, 'r', encoding='utf-8') as fa:
        reader = csv.reader(fa, delimiter='\t')
        next(reader)
        for index, row in enumerate(reader):
            assert len(row) > 8
            score = row[-1]
            sample = {'uid': cnt, 'premise': row[-3], 'hypothesis': row[-2], 'label': score}
            rows.append(sample)
            cnt += 1
    return rows


def load_ner(file, sep='\t'):
    """
    :param file:
    :param sep:
    :return:
    """
    rows = []
    sentence = []
    label = []
    uid = 0
    with open(file, 'r', encoding='utf-8') as fa:
        for line in fa:
            line = line.strip()
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    sample = {'uid': uid, 'premise': sentence, 'label': label}
                    rows.append(sample)
                    sentence = []
                    label = []
                    uid += 1
                continue
            splits = line.split(sep)
            assert len(splits) == 2
            sentence.append(splits[0])
            label.append(splits[-1])
        if len(sentence) > 0:
            sample = {'uid': uid, 'premise': sentence, 'label': label}
            rows.append(sample)
    return rows


def load_mrc(file):
    """
    :param file:
    :return:
    """
    rows = []
    with open(file, 'r', encoding='utf-8') as fa:
        data = json.load(fa)
    for paragraph in data:
        context = paragraph['context']
        uid = paragraph['id']
        question = paragraph['query']
        label = paragraph['label']
        start_position = paragraph['start_position']
        if len(start_position) > 0:
            answer_start = start_position
            answer_end = paragraph['end_position']
        else:
            answer_start = -1
            answer_end = -1
        sample = {'uid': uid, 'premise': context, 'hypothesis': question,
                  'label': "%s:::%s:::%s" % (answer_start, answer_end, label)}
        rows.append(sample)
    return rows


def load_one_premise_and_one_sequence(file, sep='\t'):
    """
    :param file:
    :param sep:
    :return:
    """
    rows = []
    sentence = []
    cls_label = []
    ner_label = []
    uid = 0
    with open(file, 'r', encoding='utf-8') as fa:
        for line in fa:
            line = line.strip()
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    sample = {'uid': uid, 'premise': sentence, 'bcls_label': cls_label, 'ner_label': ner_label}
                    rows.append(sample)
                    sentence = []
                    ner_label = []
                    uid += 1
                continue
            splits = line.split(sep)
            assert len(splits) == 3
            sentence.append(splits[0])
            ner_label.append(splits[1])
            cls_label = splits[-1]
        if len(sentence) > 0:
            sample = {'uid': uid, 'premise': sentence, 'bcls_label': cls_label, 'label': ner_label}
            rows.append(sample)
    return rows


def dump_PremiseOnly(rows, output_path):
    """
    for single-label and single-sentence classification
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'label', 'premise']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(row[col])
            fa.write('\t'.join(row_str) + '\n')


def dump_PremiseAndOneHypothesis(rows, output_path):
    """
    for single NLI classification and STS task
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'label', 'premise', 'hypothesis']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(str(row[col]))
            fa.write('\t'.join(row_str) + '\n')


def dump_PremiseAndMultiHypothesis(rows, output_path):
    """
    for multi-NLI classification
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'label', 'premise']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(str(row[col]))
            hypothesis = row['hypothesis']
            for idx, one_hypo in enumerate(hypothesis):
                if '\t' in str(one_hypo):
                    hypothesis[idx] = one_hypo.replace('\t', ' ')
                    logger.warning('%s:%s:hypothesis has tab' % (output_path, index))
            row_str.append('\t'.join(hypothesis))
            fa.write('\t'.join(row_str) + '\n')


def dump_Sequence(rows, output_path):
    """
    for ner task
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            if '\t' in str(row['uid']):
                row['uid'] = row['uid'].replace('\t', ' ')
                logger.warning('%s:%s:%s has tab' % (output_path, index, 'uid'))
            row_str.append(str(row['uid']))
            for col in ['label', 'premise']:
                for idx, token in enumerate(row[col]):
                    if '\t' in str(token):
                        row[col][idx] = token.replace('\t', ' ')
                        logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(json.dumps(row[col], ensure_ascii=False))
                # dumps的结果为string
            fa.write('\t'.join(row_str) + '\n')


def dump_Mrc(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for i, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'label', 'premise', 'hypothesis']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s: %s has tab' % (output_path, i, col))
                row_str.append(str(row[col]))
            fa.write('\t'.join(row_str) + '\n')


def dump_OnePremiseAndOneSequence(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'bcls_label']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(str(row[col]))

            for col in ['ner_label', 'premise']:
                for idx, token in enumerate(row[col]):
                    if '\t' in str(token):
                        row[col][idx] = token.replace('\t', ' ')
                        logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(json.dumps(row[col], ensure_ascii=False))
            fa.write('\t'.join(row_str) + '\n')


def load_one_sequence_and_one_sequence(file, sep='\t'):
    """
    :param file:
    :param sep:
    :return:
    """
    rows = []
    sentence = []
    ner_label = []
    mt_label = []
    uid = 0
    with open(file, 'r', encoding='utf-8') as fa:
        for line in fa:
            line = line.strip()
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'mt_label': mt_label}
                    rows.append(sample)
                    sentence = []
                    ner_label = []
                    mt_label = []
                    uid += 1
                continue
            splits = line.split(sep)
            assert len(splits) == 3
            sentence.append(splits[0])
            ner_label.append(splits[1])
            mt_label.append(splits[-1])
        if len(sentence) > 0:
            sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'mt_label': mt_label}
            rows.append(sample)
    return rows


def load_one_premise_and_two_sequence(file, sep='\t'):
    """
    :param file:
    :param sep:
    :return:
    """
    rows = []
    sentence = []
    ner_label = []
    mt_label = []
    cls_label = []
    uid = 0
    with open(file, 'r', encoding='utf-8') as fa:
        for line in fa:
            line = line.strip()
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'mt_label': mt_label, 'cls_label': cls_label}
                    rows.append(sample)
                    sentence = []
                    ner_label = []
                    mt_label = []
                    uid += 1
                continue
            splits = line.split(sep)
            assert len(splits) == 4
            sentence.append(splits[0])
            ner_label.append(splits[1])
            mt_label.append(splits[-1])
            cls_label = splits[2]
        if len(sentence) > 0:
            sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'mt_label': mt_label, 'cls_label': cls_label}
            rows.append(sample)
    return rows


def dump_OneSequenceAndOneSequence(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            if '\t' in str(row['uid']):
                row['uid'] = row['uid'].replace('\t', ' ')
                logger.warning('%s:%s:%s has tab' % (output_path, index, 'uid'))
            row_str.append(str(row['uid']))
            for col in ['ner_label', 'mt_label', 'premise']:
                for idx, token in enumerate(row[col]):
                    if '\t' in str(token):
                        row[col][idx] = token.replace('\t', ' ')
                        logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(json.dumps(row[col], ensure_ascii=False))
                # dumps的结果为string
            fa.write('\t'.join(row_str) + '\n')


def dump_OnePremiseAndTwoSequence(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'cls_label']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(str(row[col]))
            for col in ['ner_label', 'mt_label', 'premise']:
                for idx, token in enumerate(row[col]):
                    if '\t' in str(token):
                        row[col][idx] = token.replace('\t', ' ')
                        logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(json.dumps(row[col], ensure_ascii=False))
                # dumps的结果为string
            fa.write('\t'.join(row_str) + '\n')


def load_two_premise_and_one_sequence(file, sep='\t'):
    """
    :param file:
    :param sep:
    :return:
    """
    rows = []
    sentence = []
    ner_label = []
    uid = 0
    with open(file, 'r', encoding='utf-8') as fa:
        for line in fa:
            line = line.strip()
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'bcls_label': bcls_label, 'mcls_label': mcls_label}
                    rows.append(sample)
                    sentence = []
                    ner_label = []
                    uid += 1
                continue
            splits = line.split(sep)
            assert len(splits) == 4
            sentence.append(splits[0])
            ner_label.append(splits[1])
            bcls_label = splits[2]
            mcls_label = splits[3]
        if len(sentence) > 0:
            sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'bcls_label': bcls_label, 'mcls_label': mcls_label}
            rows.append(sample)
    return rows


def load_two_premise_and_two_sequence(file, sep='\t'):
    """
    :param file:
    :param sep:
    :return:
    """
    rows = []
    sentence = []
    ner_label = []
    mt_label = []
    uid = 0
    with open(file, 'r', encoding='utf-8') as fa:
        for line in fa:
            line = line.strip()
            if len(line) == 0 or line[0] == '\n':
                if len(sentence) > 0:
                    sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'mt_label': mt_label, 'bcls_label': bcls_label, 'mcls_label': mcls_label}
                    rows.append(sample)
                    sentence = []
                    ner_label = []
                    mt_label = []
                    uid += 1
                continue
            splits = line.split(sep)
            assert len(splits) == 5
            sentence.append(splits[0])
            ner_label.append(splits[1])
            mt_label.append(splits[4])
            bcls_label = splits[2]
            mcls_label = splits[3]
        if len(sentence) > 0:
            sample = {'uid': uid, 'premise': sentence, 'ner_label': ner_label, 'mt_label': mt_label, 'bcls_label': bcls_label, 'mcls_label': mcls_label}
            rows.append(sample)
    return rows


def dump_TwoPremiseAndOneSequence(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'bcls_label', 'mcls_label']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(str(row[col]))
            for col in ['ner_label', 'premise']:
                for idx, token in enumerate(row[col]):
                    if '\t' in str(token):
                        row[col][idx] = token.replace('\t', ' ')
                        logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(json.dumps(row[col], ensure_ascii=False))
                # dumps的结果为string
            fa.write('\t'.join(row_str) + '\n')


def dump_TwoPremiseAndTwoSequence(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'bcls_label', 'mcls_label']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(str(row[col]))
            for col in ['ner_label', 'mt_label', 'premise']:
                for idx, token in enumerate(row[col]):
                    if '\t' in str(token):
                        row[col][idx] = token.replace('\t', ' ')
                        logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(json.dumps(row[col], ensure_ascii=False))
                # dumps的结果为string
            fa.write('\t'.join(row_str) + '\n')


def load_two_premise(file):
    """
    :param file:
    :return:
    """
    rows = []
    with open(file, encoding='utf-8') as fa:
        line_list = fa.readlines()
        for index, row in enumerate(line_list):
            row = row.strip().split('\t')
            assert len(row) == 3
            # row[0]=label, row[1]=premise
            bcls_label = row[0]
            mcls_label = row[1]
            sample = {'uid': str(index), 'premise': row[-1], 'bcls_label': bcls_label, 'mcls_label': mcls_label}
            rows.append(sample)
    return rows


def dump_TwoPremise(rows, output_path):
    """
    :param rows:
    :param output_path:
    :return:
    """
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as fa:
        for index, row in enumerate(rows):
            row_str = []
            for col in ['uid', 'bcls_label', 'mcls_label', 'premise']:
                if '\t' in str(row[col]):
                    row[col] = row[col].replace('\t', ' ')
                    logger.warning('%s:%s:%s has tab' % (output_path, index, col))
                row_str.append(row[col])
            fa.write('\t'.join(row_str) + '\n')

def dump_rows(rows, output_path, data_format: DataFormat):
    """
    output files should have following data format
    :param rows:
    :param output_path:
    :param data_format:
    :return:
    """
    if data_format == DataFormat.PremiseOnly:
        dump_PremiseOnly(rows, output_path)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        dump_PremiseAndOneHypothesis(rows, output_path)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        dump_PremiseAndMultiHypothesis(rows, output_path)
    elif data_format == DataFormat.Sequence:
        dump_Sequence(rows, output_path)
    elif data_format == DataFormat.MRC:
        dump_Mrc(rows, output_path)
    elif data_format == DataFormat.OnePremiseAndOneSequence:
        dump_OnePremiseAndOneSequence(rows, output_path)
    elif data_format == DataFormat.OneSequenceAndOneSequence:
        dump_OneSequenceAndOneSequence(rows, output_path)
    elif data_format == DataFormat.OnePremiseAndTwoSequence:
        dump_OnePremiseAndTwoSequence(rows, output_path)
    elif data_format == DataFormat.TwoPremiseAndOneSequence:
        dump_TwoPremiseAndOneSequence(rows, output_path)
    elif data_format == DataFormat.TwoPremiseAndTwoSequence:
        dump_TwoPremiseAndTwoSequence(rows, output_path)
    elif data_format == DataFormat.TwoPremise:
        dump_TwoPremise(rows, output_path)
    else:
        raise ValueError(data_format)


if __name__ == '__main__':
    x = load_mrc('NCBI-disease-MRC/test.json')
    print(x)
    dump_Mrc(x, 'test.tsv')

