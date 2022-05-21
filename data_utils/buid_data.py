"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-29
"""
import torch
import json
from data_utils.task_def import TaskType, DataFormat, EncoderModelType
import logging
from data_utils import mrc_utils
MAX_SEQ_LEN = 256


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    优先截断长的句子
    :param tokens_a:
    :param tokens_b:
    :param max_length:
    :return:
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def bert_feature_extractor(text_a, text_b=None, max_seq_length=MAX_SEQ_LEN, tokenizer_fn=None):
    """
    :param text_a:
    :param text_b:
    :param max_seq_length:
    :param tokenizer_fn:
    :return:
    """
    logger = logging.getLogger(__name__)
    # wordpiece分词
    tokens_a = tokenizer_fn.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer_fn.tokenize(text_b)

    if tokens_b:
        # Account for [cls], [sep], [sep] with '-3'
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [cls[, [sep] with '-2'
        if len(tokens_a) > max_seq_length - 2:
            logger.debug('%s: longer than %s', text_a, max_seq_length)
            tokens_a = tokens_a[:max_seq_length - 2]

    if tokens_b:
        # Converts a sequence of tokens into ids using the vocab.
        input_ids = tokenizer_fn.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]'])
        # 2 for [cls] [sep] 1 for [sep]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    else:
        input_ids = tokenizer_fn.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * len(input_ids)
    input_mask = None
    return input_ids, input_mask, segment_ids


def build_data_premise_only(data, dump_path, max_seq_length=MAX_SEQ_LEN, tokenizer=None,
                            encoderModelType=EncoderModelType.BERT):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            label = sample['label']
            if encoderModelType == EncoderModelType.BERT:
                input_ids, _, type_ids = bert_feature_extractor(premise, max_seq_length=max_seq_length,
                                                                tokenizer_fn=tokenizer)
                features = {'uid': idx, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                fa.write('{}\n'.format(json.dumps(features)))
            else:
                raise ValueError


def build_data_premise_and_one_hypo(data, dump_path, task_type, max_seq_length=MAX_SEQ_LEN, tokenizer=None,
                                    encoderModelType=EncoderModelType.BERT):
    """
    Build data of sentence-pair tasks
    :param data:
    :param dump_path:
    :param task_type:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['label']
            if encoderModelType == EncoderModelType.BERT:
                input_ids, _, type_ids = bert_feature_extractor(
                    premise, hypothesis, max_seq_length=max_seq_length, tokenizer_fn=tokenizer
                )
                if task_type == TaskType.Span:
                    # Todo for reading comprehension and question answering
                    pass
                else:
                    features = {
                        'uid': idx,
                        'label': label,
                        'token_id': input_ids,
                        'type_id': type_ids
                    }
                fa.write('{}\n'.format(json.dumps(features)))


def build_data_premise_and_multi_hypo(data, dump_path, max_seq_length=MAX_SEQ_LEN, tokenizer=None,
                                      encoderModelType=EncoderModelType.BERT):
    """
    Build QNLI as a pair-wise ranking task
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for idx, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            hypothesis_1 = sample['hypothesis'][0]
            hypothesis_2 = sample['hypothesis'][1]
            label = sample['label']
            if encoderModelType == EncoderModelType.BERT:
                input_idx_1, _, type_idx_1 = bert_feature_extractor(
                    premise, hypothesis_1, max_seq_length=max_seq_length, tokenizer_fn=tokenizer
                )
                input_idx_2, _, type_idx_2 = bert_feature_extractor(
                    premise, hypothesis_2, max_seq_length=max_seq_length, tokenizer_fn=tokenizer
                )
                features = {
                    'uid': idx, 'label': label,
                    'token_id': [input_idx_1, input_idx_2],
                    'type_id': [type_idx_1, type_idx_2],
                    'ruid': sample['ruid'],
                    'olabel': sample['olabel'],
                }
                fa.write('{}\n'.format(json.dumps(features)))


def build_data_sequence(data, dump_path, max_seq_length=MAX_SEQ_LEN, tokenizer=None, lab_mapper=None):
    """
    build data for sequence labeling task
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param lab_mapper:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            ne_labels = sample['label']

            tokens = []
            labels = []
            for idy, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        labels.append(ne_labels[idy])
                    else:
                        labels.append(lab_mapper['X'])

            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            type_ids = [0] * len(input_ids)
            label = [lab_mapper['CLS']] + labels + [lab_mapper['SEP']]
            assert len(label) == len(input_ids)
            features = {
                'uid': idx,
                'label': label,
                'token_id': input_ids,
                'type_id': type_ids
            }
            fa.write('{}\n'.format(json.dumps(features)))


def build_data_mrc(data, dump_path, max_seq_length, tokenizer, encoderModelType):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            ids = sample['uid']
            doc = sample['premise']
            query = sample['hypothesis']
            label = sample['label']
            doc_tokens, cw_map = mrc_utils.token_doc(doc)
            answer_start, answer_end, entity_type, answer = mrc_utils.parse_mrc_label(label, doc)
            # is_vaild = mrc_utils.is_vaild_answer(doc_tokens, answer_start_adjusted, answer_end_adjusted, answer)
            # for english
            words = doc.split()
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in answer_start]
            end_positions = [x + sum([len(w) for w in words[:x+1]]) for x in answer_end]

            feature_list = []
            query_context_tokens = tokenizer.encode(query, doc, add_special_tokens=True)
            tokens = query_context_tokens.ids
            # print(tokens) list
            type_ids = query_context_tokens.type_ids
            # print(type_ids) list
            offsets = query_context_tokens.offsets
            # print(offsets) list
            # find new start_positions / end_positions, considering
            # 1. we add query tokens at begining
            # 2. word-piece tokenize
            origin_offset2token_idx_start = {}
            origin_offset2token_idx_end = {}
            for token_idx in range(len(tokens)):
                # skip query tokens
                if type_ids[token_idx] == 0:
                    continue
                token_start, token_end = offsets[token_idx]
                # skip [CLS] or [SEP]
                if token_start == token_end == 0:
                    continue
                origin_offset2token_idx_start[token_start] = token_idx
                origin_offset2token_idx_end[token_end] = token_idx

            new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
            new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

            label_mask = [
                (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
                for token_idx in range(len(tokens))
            ]
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()

            # the start/end position must be whole word
            for token_idx in range(len(tokens)):
                current_word_idx = query_context_tokens.words[token_idx]
                next_word_idx = query_context_tokens.words[token_idx + 1] if token_idx + 1 < len(tokens) else None
                prev_word_idx = query_context_tokens.words[token_idx - 1] if token_idx - 1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

            assert all(start_label_mask[p] != 0 for p in new_start_positions)
            assert all(end_label_mask[p] != 0 for p in new_end_positions)

            assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
            assert len(label_mask) == len(tokens)

            start_labels = [(1 if idx in new_start_positions else 0)
                            for idx in range(len(tokens))]
            end_labels = [(1 if idx in new_end_positions else 0)
                          for idx in range(len(tokens))]

            # truncate
            tokens = tokens[: max_seq_length]
            type_ids = type_ids[: max_seq_length]
            start_labels = start_labels[: max_seq_length]
            end_labels = end_labels[: max_seq_length]
            start_label_mask = start_label_mask[: max_seq_length]
            end_label_mask = end_label_mask[: max_seq_length]

            # make sure last token is [SEP]
            sep_token = tokenizer.token_to_id("[SEP]")
            if tokens[-1] != sep_token:
                assert len(tokens) == max_seq_length
                tokens = tokens[: -1] + [sep_token]
                start_labels[-1] = 0
                end_labels[-1] = 0
                start_label_mask[-1] = 0
                end_label_mask[-1] = 0

            seq_len = len(tokens)
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)

            for start, end in zip(new_start_positions, new_end_positions):
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            f = json.dumps({
                'uid': ids,
                'token_id': tokens,
                'type_id': type_ids,
                'start_position': start_labels,
                'end_position': end_labels,
                'start_position_mask': start_label_mask,
                'end_position_mask': end_label_mask,
                'match_labels': match_labels.tolist(),
                'label': entity_type,
                'doc': doc,
                'answer': [answer]
            })
            fa.write('{}\n'.format(f))


def build_data_joint_two(data, dump_path, max_seq_length, tokenizer, encoderModelType, ner_lab_dict):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :param ner_lab_dict:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            ne_labels = sample['ner_label']
            cls_label = sample['cls_label']
            tokens = []
            ner_labels = []
            for idy, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        ner_labels.append(ne_labels[idy])
                    else:
                        ner_labels.append(ner_lab_dict['X'])

            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            type_ids = [0] * len(input_ids)
            ner_label = [ner_lab_dict['CLS']] + ner_labels + [ner_lab_dict['SEP']]
            assert len(ner_label) == len(input_ids)
            features = {
                'uid': idx,
                'cls_label': cls_label,
                'ner_label': ner_label,
                'token_id': input_ids,
                'type_id': type_ids
            }
            fa.write('{}\n'.format(json.dumps(features)))


def build_data_joint_mt(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_a, lab_dict_b):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :param lab_dict_a:
    :param lab_dict_b:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            mt_labels = sample['mt_label']
            ne_labels = sample['ner_label']

            tokens = []
            mt_labels_list = []
            ner_labels_list = []
            for idy, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        ner_labels_list.append(ne_labels[idy])
                        mt_labels_list.append(mt_labels[idy])
                    else:
                        ner_labels_list.append(lab_dict_b['X'])
                        mt_labels_list.append(lab_dict_a['X'])

            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            type_ids = [0] * len(input_ids)
            mt_label = [lab_dict_a['CLS']] + mt_labels_list + [lab_dict_a['SEP']]
            ner_label = [lab_dict_b['CLS']] + ner_labels_list + [lab_dict_b['SEP']]
            assert len(ner_label) == len(input_ids) == len(mt_label)
            features = {
                'uid': idx,
                'mt_label': mt_label,
                'ner_label': ner_label,
                'token_id': input_ids,
                'type_id': type_ids
            }
            fa.write('{}\n'.format(json.dumps(features)))


def build_data_joint_three_mt(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_a, lab_dict_b):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :param lab_dict_a:
    :param lab_dict_b:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            cls_label = sample['cls_label']
            mt_labels = sample['mt_label']
            ne_labels = sample['ner_label']

            tokens = []
            mt_labels_list = []
            ner_labels_list = []
            for idy, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        ner_labels_list.append(ne_labels[idy])
                        mt_labels_list.append(mt_labels[idy])
                    else:
                        ner_labels_list.append(lab_dict_b['X'])
                        mt_labels_list.append(lab_dict_a['X'])

            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            type_ids = [0] * len(input_ids)
            mt_label = [lab_dict_a['CLS']] + mt_labels_list + [lab_dict_a['SEP']]
            ner_label = [lab_dict_b['CLS']] + ner_labels_list + [lab_dict_b['SEP']]
            assert len(ner_label) == len(input_ids) == len(mt_label)
            features = {
                'uid': idx,
                'cls_label': cls_label,
                'mt_label': mt_label,
                'ner_label': ner_label,
                'token_id': input_ids,
                'type_id': type_ids
            }
            fa.write('{}\n'.format(json.dumps(features)))


def build_data_joint_three_cls(data, dump_path, max_seq_length, tokenizer, encoderModelType, ner_lab_dict):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :param ner_lab_dict:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            bcls_label = sample['bcls_label']
            mcls_label = sample['mcls_label']
            ne_labels = sample['ner_label']

            tokens = []
            ner_labels_list = []
            for idy, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        ner_labels_list.append(ne_labels[idy])
                    else:
                        ner_labels_list.append(ner_lab_dict['X'])

            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            type_ids = [0] * len(input_ids)
            ner_label = [ner_lab_dict['CLS']] + ner_labels_list + [ner_lab_dict['SEP']]
            assert len(ner_label) == len(input_ids)
            features = {
                'uid': idx,
                'bcls_label': bcls_label,
                'mcls_label': mcls_label,
                'ner_label': ner_label,
                'token_id': input_ids,
                'type_id': type_ids
            }
            fa.write('{}\n'.format(json.dumps(features)))


def build_data_joint_all_cls(data, dump_path, max_seq_length, tokenizer, encoderModelType, label_mapper_ner, label_mapper_mt):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :param label_mapper_ner:
    :param label_mapper_mt:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            bcls_label = sample['bcls_label']
            mcls_label = sample['mcls_label']
            mt_labels = sample['mt_label']
            ne_labels = sample['ner_label']

            tokens = []
            mt_labels_list = []
            ner_labels_list = []
            for idy, word in enumerate(premise):
                subwords = tokenizer.tokenize(word)
                # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        ner_labels_list.append(ne_labels[idy])
                        mt_labels_list.append(mt_labels[idy])
                    else:
                        ner_labels_list.append(label_mapper_ner['X'])
                        mt_labels_list.append(label_mapper_mt['X'])

            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
            type_ids = [0] * len(input_ids)
            mt_label = [label_mapper_mt['CLS']] + mt_labels_list + [label_mapper_mt['SEP']]
            ner_label = [label_mapper_ner['CLS']] + ner_labels_list + [label_mapper_ner['SEP']]
            assert len(ner_label) == len(input_ids) == len(mt_label)
            features = {
                'uid': idx,
                'bcls_label': bcls_label,
                'mcls_label': mcls_label,
                'mt_label': mt_label,
                'ner_label': ner_label,
                'token_id': input_ids,
                'type_id': type_ids
            }
            fa.write('{}\n'.format(json.dumps(features)))


def build_data_joint_bcls_mcls(data, dump_path, max_seq_length, tokenizer, encoderModelType, lab_dict_a, lab_dict_b):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param encoderModelType:
    :param lab_dict_a:
    :param lab_dict_b:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            premise = sample['premise']
            bcls_label = sample['bcls_label']
            mcls_label = sample['mcls_label']
            if encoderModelType == EncoderModelType.BERT:
                input_ids, _, type_ids = bert_feature_extractor(premise, max_seq_length=max_seq_length,
                                                                tokenizer_fn=tokenizer)
                features = {'uid': idx, 'bcls_label': bcls_label, 'mcls_label': mcls_label, 'token_id': input_ids,
                            'type_id': type_ids}
                fa.write('{}\n'.format(json.dumps(features)))
            else:
                raise ValueError


def build_data_mrc_ner(data, dump_path, max_seq_length, tokenizer, mrc_tokenizer, encoderModelType, lab_dict):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param mrc_tokenizer:
    :param encoderModelType:
    :param lab_dict:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            domain_label = sample['domain_label']
            # premise是正文, hypothesis是query, ner_label是ner标签
            query = sample['hypothesis']
            context = sample['premise']
            start_positions = eval(sample['start_position'])
            end_positions = eval(sample['end_position'])
            span_positions = eval(sample['span_position'])
            ne_labels = sample['ner_label']
            query_list = query.split()
            context_list = context.split()
            query_ner_label_list = []
            context_ner_label_list = []
            query_tokens = []
            context_tokens = []
            for idy, word in enumerate(query_list):
                subwords = tokenizer.tokenize(word)
                query_tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        query_ner_label_list.append(lab_dict['O'])
                    else:
                        query_ner_label_list.append(lab_dict['X'])
            for idy, word in enumerate(context_list):
                subwords = tokenizer.tokenize(word)
                context_tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        context_ner_label_list.append(ne_labels[idy])
                    else:
                        context_ner_label_list.append(lab_dict['X'])
            ner_label = [lab_dict['CLS']] + query_ner_label_list + [lab_dict['SEP']] + context_ner_label_list + [lab_dict['SEP']]
            input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + query_tokens + ['[SEP]'] + context_tokens + ['[SEP]'])
            words = context.split()
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]
            query_context_tokens = mrc_tokenizer.encode(query, context, add_special_tokens=True)
            tokens = query_context_tokens.ids
            type_ids = query_context_tokens.type_ids
            offsets = query_context_tokens.offsets
            assert input_ids == tokens
            # find new start_positions/end_positions, considering
            # 1. we add query tokens at the beginning
            # 2. word-piece tokenize
            origin_offset2token_idx_start = {}
            origin_offset2token_idx_end = {}
            for token_idx in range(len(tokens)):
                # skip query tokens
                if type_ids[token_idx] == 0:
                    continue
                token_start, token_end = offsets[token_idx]
                # skip [CLS] or [SEP]
                if token_start == token_end == 0:
                    continue
                origin_offset2token_idx_start[token_start] = token_idx
                origin_offset2token_idx_end[token_end] = token_idx
            new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
            new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
            label_mask = [
                (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
                for token_idx in range(len(tokens))
            ]
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()

            for token_idx in range(len(tokens)):
                current_word_idx = query_context_tokens.words[token_idx]
                next_word_idx = query_context_tokens.words[token_idx+1] if token_idx+1 < len(tokens) else None
                prev_word_idx = query_context_tokens.words[token_idx-1] if token_idx-1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0
            assert all(start_label_mask[p] != 0 for p in new_start_positions)
            assert all(end_label_mask[p] != 0 for p in new_end_positions)

            assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
            assert len(label_mask) == len(tokens)
            start_labels = [(1 if idx in new_start_positions else 0)
                            for idx in range(len(tokens))]
            end_labels = [(1 if idx in new_end_positions else 0)
                          for idx in range(len(tokens))]
            tokens = tokens[: max_seq_length]
            type_ids = type_ids[: max_seq_length]
            start_labels = start_labels[: max_seq_length]
            end_labels = end_labels[: max_seq_length]
            start_label_mask = start_label_mask[: max_seq_length]
            end_label_mask = end_label_mask[: max_seq_length]
            ner_label = ner_label[: max_seq_length]
            # 防止截断后最后一个token不是SEP
            sep_token = mrc_tokenizer.token_to_id("[SEP]")
            if tokens[-1] != sep_token:
                assert len(tokens) == max_seq_length
                tokens = tokens[: -1] + [sep_token]
                start_labels[-1] = 0
                end_labels[-1] = 0
                start_label_mask[-1] = 0
                end_label_mask[-1] = 0
                ner_label[-1] = 5
            # padding
            tokens = padding(tokens, 0)
            type_ids = padding(type_ids, 1)
            start_labels = padding(start_labels)
            end_labels = padding(end_labels)
            start_label_mask = padding(start_label_mask)
            end_label_mask = padding(end_label_mask)
            seq_len = len(tokens)
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in zip(new_start_positions, new_end_positions):
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1
            f = json.dumps({
                'uid': idx,
                'token_id': tokens,
                'type_id': type_ids,
                'start_position': start_labels,
                'end_position': end_labels,
                'start_position_mask': start_label_mask,
                'end_position_mask': end_label_mask,
                'domain_label': domain_label,
                'ner_label': ner_label,
            })
            fa.write('{}\n'.format(f))


def build_data_joint_all(data, dump_path, max_seq_length, tokenizer, mrc_tokenizer, encoderModelType, label_mapper_ner, label_mapper_mt):
    """
    :param data:
    :param dump_path:
    :param max_seq_length:
    :param tokenizer:
    :param mrc_tokenizer:
    :param encoderModelType:
    :param label_mapper_ner:
    :param label_mapper_mt:
    :return:
    """
    with open(dump_path, 'w', encoding='utf-8') as fa:
        for index, sample in enumerate(data):
            idx = sample['uid']
            domain_label = sample['domain_label']
            # premise是正文, hypothesis是query, ner_label是ner标签
            query = sample['hypothesis']
            context = sample['premise']
            start_positions = eval(sample['start_position'])
            end_positions = eval(sample['end_position'])
            span_positions = eval(sample['span_position'])
            mCLS_labels = sample['mCLS_label']
            bCLS_labels = sample['bCLS_label']
            mt_labels = sample['mtCLS_label']
            ne_labels = sample['ner_label']
            query_list = query.split()
            context_list = context.split()
            query_ner_label_list = []
            query_mt_label_list = []
            context_ner_label_list = []
            context_mt_label_list = []
            query_tokens = []
            context_tokens = []
            for idy, word in enumerate(query_list):
                subwords = tokenizer.tokenize(word)
                query_tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        query_ner_label_list.append(label_mapper_ner['O'])
                        query_mt_label_list.append(label_mapper_mt['O'])
                    else:
                        query_ner_label_list.append(label_mapper_ner['X'])
                        query_mt_label_list.append(label_mapper_mt['X'])

            for idy, word in enumerate(context_list):
                subwords = tokenizer.tokenize(word)
                context_tokens.extend(subwords)
                for idz in range(len(subwords)):
                    if idz == 0:
                        context_ner_label_list.append(ne_labels[idy])
                        context_mt_label_list.append(mt_labels[idy])
                    else:
                        context_ner_label_list.append(label_mapper_ner['X'])
                        context_mt_label_list.append(label_mapper_mt['X'])
            ner_label = [label_mapper_ner['CLS']] + query_ner_label_list + [label_mapper_ner['SEP']] + \
                        context_ner_label_list + [label_mapper_ner['SEP']]
            mtCLS_label = [label_mapper_mt['CLS']] + query_mt_label_list + [label_mapper_mt['SEP']] + \
                       context_mt_label_list + [label_mapper_mt['CLS']]

            input_ids = tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + query_tokens + ['[SEP]'] + context_tokens + ['[SEP]'])
            words = context.split()
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]
            query_context_tokens = mrc_tokenizer.encode(query, context, add_special_tokens=True)
            tokens = query_context_tokens.ids
            type_ids = query_context_tokens.type_ids
            offsets = query_context_tokens.offsets
            assert input_ids == tokens
            # find new start_positions/end_positions, considering
            # 1. we add query tokens at the beginning
            # 2. word-piece tokenize
            origin_offset2token_idx_start = {}
            origin_offset2token_idx_end = {}
            for token_idx in range(len(tokens)):
                # skip query tokens
                if type_ids[token_idx] == 0:
                    continue
                token_start, token_end = offsets[token_idx]
                # skip [CLS] or [SEP]
                if token_start == token_end == 0:
                    continue
                origin_offset2token_idx_start[token_start] = token_idx
                origin_offset2token_idx_end[token_end] = token_idx
            new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
            new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
            label_mask = [
                (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
                for token_idx in range(len(tokens))
            ]
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()

            for token_idx in range(len(tokens)):
                current_word_idx = query_context_tokens.words[token_idx]
                next_word_idx = query_context_tokens.words[token_idx + 1] if token_idx + 1 < len(tokens) else None
                prev_word_idx = query_context_tokens.words[token_idx - 1] if token_idx - 1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0
            assert all(start_label_mask[p] != 0 for p in new_start_positions)
            assert all(end_label_mask[p] != 0 for p in new_end_positions)

            assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
            assert len(label_mask) == len(tokens)
            start_labels = [(1 if idx in new_start_positions else 0)
                            for idx in range(len(tokens))]
            end_labels = [(1 if idx in new_end_positions else 0)
                          for idx in range(len(tokens))]
            tokens = tokens[: max_seq_length]
            type_ids = type_ids[: max_seq_length]
            start_labels = start_labels[: max_seq_length]
            end_labels = end_labels[: max_seq_length]
            start_label_mask = start_label_mask[: max_seq_length]
            end_label_mask = end_label_mask[: max_seq_length]
            ner_label = ner_label[: max_seq_length]
            mtCLS_label = mtCLS_label[: max_seq_length]

            # 防止截断后最后一个token不是SEP
            sep_token = mrc_tokenizer.token_to_id("[SEP]")
            if tokens[-1] != sep_token:
                assert len(tokens) == max_seq_length
                tokens = tokens[: -1] + [sep_token]
                start_labels[-1] = 0
                end_labels[-1] = 0
                start_label_mask[-1] = 0
                end_label_mask[-1] = 0
                ner_label[-1] = 5
                mtCLS_label[-1] = 6
            # padding
            tokens = padding(tokens, 0)
            type_ids = padding(type_ids, 1)
            start_labels = padding(start_labels)
            end_labels = padding(end_labels)
            start_label_mask = padding(start_label_mask)
            end_label_mask = padding(end_label_mask)
            seq_len = len(tokens)
            match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
            for start, end in zip(new_start_positions, new_end_positions):
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1
            f = json.dumps({
                'uid': idx,
                'token_id': tokens,
                'type_id': type_ids,
                'start_position': start_labels,
                'end_position': end_labels,
                'start_position_mask': start_label_mask,
                'end_position_mask': end_label_mask,
                'domain_label': domain_label,
                'bCLS_label': bCLS_labels,
                'mCLS_label': mCLS_labels,
                'mtCLS_label': mtCLS_label,
                'ner_label': ner_label,
            })

            fa.write('{}\n'.format(f))


def padding(lst, value=0, max_length=MAX_SEQ_LEN):
    """
    :param lst:
    :param value:
    :param max_length:
    :return:
    """
    while len(lst) < max_length:
        lst.append(value)
    return lst


def split_if_longer(data, label_mapper, max_seq_len=30):
    """
    :param data:
    :param label_mapper:
    :param max_seq_len:
    :return:
    """
    # Todo
    return data
