"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-17
"""
import sys
import json
import torch
import random
from shutil import copyfile
from data_utils.task_def import TaskType, DataFormat
from data_utils.task_def import EncoderModelType

UNK_ID=100
BOS_ID=101


class BatchGen(object):
    def __init__(self, data, batch_size=32, gpu=True, is_train=True, maxlen=256, dropout_w=0.005,
                 do_batch=True, weighted_on=False,
                 task_id=0,
                 task=None,
                 task_type=TaskType.Classification,
                 data_type=DataFormat.PremiseOnly,
                 soft_label=False,
                 encoder_type=EncoderModelType.BERT):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.is_train = is_train
        self.gpu = gpu
        self.weighted_on = weighted_on
        self.data = data
        self.task_id = task_id
        self.pairwise_size = 1
        self.data_type = data_type
        self.task_type = task_type
        self.encoder_type = encoder_type
        # soft label used for knowledge distillation
        self.soft_label_on = soft_label
        if do_batch:
            if is_train:
                indices = list(range(len(self.data)))
                random.shuffle(indices)
                data = [self.data[i] for i in indices]
            self.data = BatchGen.make_batches(data, batch_size)
        self.offset = 0
        self.dropout_w = dropout_w

    @staticmethod
    def make_batches(data, batch_size=32):
        """
        :param data:
        :param batch_size:
        :return:
        """
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    @staticmethod
    def load(path, is_train=True, maxlen=256, factor=1.0, task_type=None):
        """
        :param path:
        :param is_train:
        :param maxlen:
        :param factor:
        :param task_type:
        :return:
        """
        assert task_type is not None
        with open(path, 'r', encoding='utf-8') as fa:
            data = []
            cnt = 0
            for line in fa:
                # print(line)
                sample = json.loads(line)
                sample['factor'] = factor
                cnt += 1
                # print(len(sample['token_id']))
                if is_train:
                    if (task_type == TaskType.Ranking) and (len(sample['token_id'][0]) > maxlen or len(sample['token_id'][1] > maxlen)):
                        continue
                    if (task_type != TaskType.Ranking) and (len(sample['token_id']) > maxlen):
                        continue
                data.append(sample)
                # print(data)
            print('Loaded task_type {} {} samples out of {}'.format(task_type, len(data), cnt))
            return data

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[idx] for idx in indices]
        self.offset = 0

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else:
            return arr

    def __len__(self):
        return len(self.data)

    def patch(self, v):
        v = v.cuda(non_blocking=True)
        return v

    @staticmethod
    def todevice(v, device):
        # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        v = v.to(device)
        return v

    def rebatch(self, batch):
        new_batch = []
        for sample in batch:
            size = len(sample['token_id'])
            self.pairwise_size = size
            assert size == len(sample['type_id'])
            for idx in range(size):
                token_id = sample['token_id'][idx]
                type_id = sample['type_id'][idx]
                uid = sample['uid'][idx]
                olab = sample['olabel'][idx]
                new_batch.append({
                    'uid': uid,
                    'token_id': token_id,
                    'type_id': type_id,
                    'label': sample['label'],
                    'true_label': olab
                })
        return new_batch

    def __if__pair__(self, data_type):
        return data_type in [DataFormat.PremiseAndOneHypothesis, DataFormat.PremiseAndMultiHypothesis]

    def __iter__(self):
        # todo 简化代码
        while self.offset < len(self):
            batch = self.data[self.offset]
            if self.task_type == TaskType.Ranking:
                batch = self.rebatch(batch)

            # prepare model input, batch_info用于存放batch_data的index, 都是list
            batch_data, batch_info = self._prepare_model_input(batch)
            batch_info['task_id'] = self.task_id  # used for select correct decoding head
            batch_info['input_len'] = len(batch_data)  # used to select model inputs
            # select different loss function and other difference in training and testing
            batch_info['task_type'] = self.task_type
            batch_info['pairwise_size'] = self.pairwise_size  # needed for ranking task
            if self.gpu:
                for index, item in enumerate(batch_data):
                    # 锁页
                    batch_data[index] = self.patch(item.pin_memory())

            # add label
            if self.data_type == DataFormat.OnePremiseAndOneSequence:
                ner_labels = [sample['ner_label'] for sample in batch]
                cls_labels = [sample['cls_label'] for sample in batch]
            elif self.data_type == DataFormat.OneSequenceAndOneSequence:
                ner_labels = [sample['ner_label'] for sample in batch]
                mt_labels = [sample['mt_label'] for sample in batch]
            elif self.data_type == DataFormat.OnePremiseAndTwoSequence:
                ner_labels = [sample['ner_label'] for sample in batch]
                mt_labels = [sample['mt_label'] for sample in batch]
                cls_labels = [sample['cls_label'] for sample in batch]
            elif self.data_type == DataFormat.TwoPremiseAndOneSequence:
                ner_labels = [sample['ner_label'] for sample in batch]
                bcls_labels = [sample['bcls_label'] for sample in batch]
                mcls_labels = [sample['mcls_label'] for sample in batch]
            elif self.data_type == DataFormat.TwoPremiseAndTwoSequence:
                ner_labels = [sample['ner_label'] for sample in batch]
                mt_labels = [sample['mt_label'] for sample in batch]
                bcls_labels = [sample['bcls_label'] for sample in batch]
                mcls_labels = [sample['mcls_label'] for sample in batch]
            elif self.data_type == DataFormat.TwoPremise:
                bcls_labels = [sample['bcls_label'] for sample in batch]
                mcls_labels = [sample['mcls_label'] for sample in batch]
            else:
                labels = [sample['label'] for sample in batch]
            if self.is_train:
                # in training model, label is used by pytorch, so would be tensor
                if self.task_type == TaskType.Regression:
                    batch_data.append(torch.FloatTensor(labels))
                    batch_info['label'] = len(batch_data) - 1
                elif self.task_type in (TaskType.Classification, TaskType.Ranking):
                    batch_data.append(torch.LongTensor(labels))
                    batch_info['label'] = len(batch_data) - 1
                elif self.task_type == TaskType.Joint:
                    # add cls labels
                    batch_data.append(torch.LongTensor(cls_labels))
                    batch_info['cls_label'] = len(batch_data) - 1
                    # add ner labels
                    batch_size = self._get_batch_size(batch)
                    token_len = self.__get_max_len(batch, key='token_id')
                    # 初始化
                    tlab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(ner_labels):
                        label_length = len(label)
                        tlab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(tlab)
                    batch_info['ner_label'] = len(batch_data) - 1
                elif self.task_type == TaskType.Joint_mt:
                    # add mt labels
                    batch_size = self._get_batch_size(batch)
                    token_len = self.__get_max_len(batch, key='token_id')
                    # 初始化
                    mt_lab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(mt_labels):
                        label_length = len(label)
                        mt_lab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(mt_lab)
                    batch_info['mt_label'] = len(batch_data) - 1
                    # add ner labels
                    tlab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(ner_labels):
                        label_length = len(label)
                        tlab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(tlab)
                    batch_info['ner_label'] = len(batch_data) - 1
                elif self.task_type == TaskType.Joint_mt_three:
                    # add mt labels
                    batch_size = self._get_batch_size(batch)
                    token_len = self.__get_max_len(batch, key='token_id')
                    # 初始化
                    mt_lab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(mt_labels):
                        label_length = len(label)
                        mt_lab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(mt_lab)
                    batch_info['mt_label'] = len(batch_data) - 1
                    # add ner labels
                    tlab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(ner_labels):
                        label_length = len(label)
                        tlab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(tlab)
                    batch_info['ner_label'] = len(batch_data) - 1
                    # add cls labels
                    batch_data.append(torch.LongTensor(cls_labels))
                    batch_info['cls_label'] = len(batch_data) - 1
                elif self.task_type == TaskType.Joint_cls_three:
                    # add bcls labels
                    batch_data.append(torch.LongTensor(bcls_labels))
                    batch_info['bcls_label'] = len(batch_data) - 1
                    # add mcls labels
                    batch_data.append(torch.LongTensor(mcls_labels))
                    batch_info['mcls_label'] = len(batch_data) - 1
                    # add ner labels
                    batch_size = self._get_batch_size(batch)
                    token_len = self.__get_max_len(batch, key='token_id')
                    # 初始化
                    tlab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(ner_labels):
                        label_length = len(label)
                        tlab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(tlab)
                    batch_info['ner_label'] = len(batch_data) - 1
                elif self.task_type == TaskType.Joint_all:
                    # add mt labels
                    batch_size = self._get_batch_size(batch)
                    token_len = self.__get_max_len(batch, key='token_id')
                    # 初始化
                    mt_lab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(mt_labels):
                        label_length = len(label)
                        mt_lab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(mt_lab)
                    batch_info['mt_label'] = len(batch_data) - 1
                    # add ner labels
                    tlab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(ner_labels):
                        label_length = len(label)
                        tlab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(tlab)
                    batch_info['ner_label'] = len(batch_data) - 1
                    # add bcls labels
                    batch_data.append(torch.LongTensor(bcls_labels))
                    batch_info['bcls_label'] = len(batch_data) - 1
                    # add mcls labels
                    batch_data.append(torch.LongTensor(mcls_labels))
                    batch_info['mcls_label'] = len(batch_data) - 1
                elif self.task_type == TaskType.Joint_bCLS_mCLS:
                    # add bcls labels
                    batch_data.append(torch.LongTensor(bcls_labels))
                    batch_info['bcls_label'] = len(batch_data) - 1
                    # add mcls labels
                    batch_data.append(torch.LongTensor(mcls_labels))
                    batch_info['mcls_label'] = len(batch_data) - 1
                elif self.task_type == TaskType.ReadingComprehension:
                    token_len = self.__get_max_len(batch, key='token_id')
                    start_labels_list = []
                    end_labels_list = []
                    start_labels_mask_list = []
                    end_labels_mask_list = []
                    match_labels_list = []
                    for sample in batch:
                        temp_start_labels_list = sample['start_position']
                        if len(temp_start_labels_list) <= token_len:
                            padding_length = token_len - len(temp_start_labels_list)
                            temp_start_labels_list += [0] * padding_length
                            start_labels_list.append(temp_start_labels_list)
                        else:
                            raise ValueError
                        temp_end_labels_list = sample['end_position']
                        if len(temp_end_labels_list) <= token_len:
                            padding_length = token_len - len(temp_end_labels_list)
                            temp_end_labels_list += [0] * padding_length
                            end_labels_list.append(temp_end_labels_list)
                        else:
                            raise ValueError
                        temp_start_labels_mask_list = sample['start_position_mask']
                        if len(temp_start_labels_mask_list) <= token_len:
                            padding_length = token_len - len(temp_start_labels_mask_list)
                            temp_start_labels_mask_list += [0] * padding_length
                            start_labels_mask_list.append(temp_start_labels_mask_list)
                        else:
                            raise ValueError
                        temp_end_labels_mask_list = sample['end_position_mask']
                        if len(temp_end_labels_mask_list) <= token_len:
                            padding_length = token_len - len(temp_end_labels_mask_list)
                            temp_end_labels_mask_list += [0] * padding_length
                            end_labels_mask_list.append(temp_end_labels_mask_list)
                        else:
                            raise ValueError
                        temp_match_labels_list = sample['match_labels']
                        assert len(temp_match_labels_list) == len(temp_match_labels_list[0])
                        if len(temp_match_labels_list) <= token_len:
                            padding_length = token_len - len(temp_match_labels_list)
                            temp_match_labels_list += [[]] * padding_length
                            for idx in range(len(temp_match_labels_list)):
                                temp_padding_length = token_len - len(temp_match_labels_list[idx])
                                temp_match_labels_list[idx] += [0] * temp_padding_length
                            assert len(temp_match_labels_list) == len(temp_match_labels_list[-1])
                            match_labels_list.append(temp_match_labels_list)
                        else:
                            raise ValueError

                    start_labels = torch.LongTensor(start_labels_list)
                    batch_data.append(start_labels)
                    batch_info['start_labels'] = len(batch_data) - 1
                    end_labels = torch.LongTensor(end_labels_list)
                    batch_data.append(end_labels)
                    batch_info['end_labels'] = len(batch_data) - 1
                    start_label_mask = torch.LongTensor(start_labels_mask_list)
                    batch_data.append(start_label_mask)
                    batch_info['start_label_mask'] = len(batch_data) - 1
                    end_label_mask = torch.LongTensor(end_labels_mask_list)
                    batch_data.append(end_label_mask)
                    batch_info['end_label_mask'] = len(batch_data) - 1
                    match_labels = torch.LongTensor(match_labels_list)
                    batch_data.append(match_labels)
                    batch_info['match_labels'] = len(batch_data) - 1
                    batch_data.append(sample['answer'] for sample in batch)
                    batch_info['label'] = len(batch_data) - 1
                elif self.task_type == TaskType.SequenceLabeling:
                    batch_size = self._get_batch_size(batch)
                    token_len = self.__get_max_len(batch, key='token_id')
                    # 初始化
                    tlab = torch.LongTensor(batch_size, token_len).fill_(-1)
                    for index, label in enumerate(labels):
                        label_length = len(label)
                        tlab[index, :label_length] = torch.LongTensor(label)
                    batch_data.append(tlab)
                    batch_info['label'] = len(batch_data) - 1
                else:
                    raise ValueError

                # soft label generated by ensemble models for knowledge distillation
                if self.soft_label_on and (batch[0].get('softlabel', None) is not None):
                    assert self.task_type != TaskType.Span  # Span task doesn't support soft label yet
                    softlabels = [sample['softlabel'] for sample in batch]
                    softlabels = torch.FloatTensor(softlabels)
                    batch_info['soft_label'] = self.patch(softlabels.pin_memory()) if self.gpu else softlabels
            else:
                # in test model, label would be used for evaluation
                if self.data_type == DataFormat.OnePremiseAndOneSequence:
                    batch_info['ner_label'] = ner_labels
                    batch_info['cls_label'] = cls_labels
                elif self.data_type == DataFormat.OneSequenceAndOneSequence:
                    batch_info['ner_label'] = ner_labels
                    batch_info['mt_label'] = mt_labels
                elif self.data_type == DataFormat.OnePremiseAndTwoSequence:
                    batch_info['ner_label'] = ner_labels
                    batch_info['mt_label'] = mt_labels
                    batch_info['cls_label'] = cls_labels
                elif self.data_type == DataFormat.TwoPremiseAndOneSequence:
                    batch_info['ner_label'] = ner_labels
                    batch_info['bcls_label'] = bcls_labels
                    batch_info['mcls_label'] = mcls_labels
                elif self.data_type == DataFormat.TwoPremiseAndTwoSequence:
                    batch_info['ner_label'] = ner_labels
                    batch_info['mt_label'] = mt_labels
                    batch_info['bcls_label'] = bcls_labels
                    batch_info['mcls_label'] = mcls_labels
                elif self.data_type == DataFormat.TwoPremise:
                    batch_info['bcls_label'] = bcls_labels
                    batch_info['mcls_label'] = mcls_labels
                else:
                    batch_info['label'] = labels

                if self.task_type == TaskType.Ranking:
                    batch_info['true_label'] = [sample['true_label'] for sample in batch]
                if self.task_type == TaskType.ReadingComprehension:
                    token_len = self.__get_max_len(batch, key='token_id')
                    start_labels_list = []
                    end_labels_list = []
                    start_labels_mask_list = []
                    end_labels_mask_list = []
                    match_labels_list = []
                    for sample in batch:
                        temp_start_labels_list = sample['start_position']
                        if len(temp_start_labels_list) <= token_len:
                            padding_length = token_len - len(temp_start_labels_list)
                            temp_start_labels_list += [0] * padding_length
                            start_labels_list.append(temp_start_labels_list)
                        else:
                            raise ValueError
                        temp_end_labels_list = sample['end_position']
                        if len(temp_end_labels_list) <= token_len:
                            padding_length = token_len - len(temp_end_labels_list)
                            temp_end_labels_list += [0] * padding_length
                            end_labels_list.append(temp_end_labels_list)
                        else:
                            raise ValueError
                        temp_start_labels_mask_list = sample['start_position_mask']
                        if len(temp_start_labels_mask_list) <= token_len:
                            padding_length = token_len - len(temp_start_labels_mask_list)
                            temp_start_labels_mask_list += [0] * padding_length
                            start_labels_mask_list.append(temp_start_labels_mask_list)
                        else:
                            raise ValueError
                        temp_end_labels_mask_list = sample['end_position_mask']
                        if len(temp_end_labels_mask_list) <= token_len:
                            padding_length = token_len - len(temp_end_labels_mask_list)
                            temp_end_labels_mask_list += [0] * padding_length
                            end_labels_mask_list.append(temp_end_labels_mask_list)
                        else:
                            raise ValueError
                        temp_match_labels_list = sample['match_labels']
                        assert len(temp_match_labels_list) == len(temp_match_labels_list[0])
                        if len(temp_match_labels_list) <= token_len:
                            padding_length = token_len - len(temp_match_labels_list)
                            temp_match_labels_list += [[]] * padding_length
                            for idx in range(len(temp_match_labels_list)):
                                temp_padding_length = token_len - len(temp_match_labels_list[idx])
                                temp_match_labels_list[idx] += [0] * temp_padding_length
                            assert len(temp_match_labels_list) == len(temp_match_labels_list[-1])
                            match_labels_list.append(temp_match_labels_list)
                        else:
                            raise ValueError

                    start_labels = torch.LongTensor(start_labels_list)
                    batch_data.append(start_labels)
                    batch_info['start_labels'] = len(batch_data) - 1
                    end_labels = torch.LongTensor(end_labels_list)
                    batch_data.append(end_labels)
                    batch_info['end_labels'] = len(batch_data) - 1
                    start_label_mask = torch.LongTensor(start_labels_mask_list)
                    batch_data.append(start_label_mask)
                    batch_info['start_label_mask'] = len(batch_data) - 1
                    end_label_mask = torch.LongTensor(end_labels_mask_list)
                    batch_data.append(end_label_mask)
                    batch_info['end_label_mask'] = len(batch_data) - 1
                    match_labels = torch.LongTensor(match_labels_list)
                    batch_data.append(match_labels)
                    batch_info['match_labels'] = len(batch_data) - 1
                    batch_data.append(sample['answer'] for sample in batch)
                    batch_info['label'] = len(batch_data) - 1
                    batch_info['doc'] = [sample['doc'] for sample in batch]
                    # batch_info['tokens'] = [sample['tokens'] for sample in batch]
                    batch_info['answer'] = [sample['answer'] for sample in batch]


            batch_info['uids'] = [sample['uid'] for sample in batch] # used in scoring
            self.offset += 1
            yield batch_info, batch_data

    def __get_max_len(self, batch, key='token_id'):
        # token_len = max(len(x[key]) for x in batch)
        token_len = self.maxlen
        return token_len

    def _get_batch_size(self,batch):
        return len(batch)

    def _prepare_model_input(self, batch):
        batch_size = self._get_batch_size(batch)
        tok_len = self.__get_max_len(batch)
        # Todo check it
        hypothesis_len = max(len(x['type_id']) - sum(x['type_id']) for x in batch)
        if self.encoder_type == EncoderModelType.ROBERTA:
            # 用全1填充
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(1)
            # 用全0填充
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)
        else:
            token_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            type_ids = torch.LongTensor(batch_size, tok_len).fill_(0)
            masks = torch.LongTensor(batch_size, tok_len).fill_(0)

        if self.__if__pair__(self.data_type):
            premise_masks = torch.ByteTensor(batch_size, tok_len).fill_(1)
            hypothesis_masks = torch.ByteTensor(batch_size, hypothesis_len).fill_(1)

        for index, sample in enumerate(batch):
            select_len = min(len(sample['token_id']), tok_len)
            tok = sample['token_id']
            if self.is_train:
                tok = self.__random_select__(tok)
            # 对第index行进行裁剪
            token_ids[index, :select_len] = torch.LongTensor(tok[:select_len])
            type_ids[index, :select_len] = torch.LongTensor(sample['type_id'][:select_len])
            masks[index, :select_len] = torch.LongTensor([1] * select_len)
            if self.__if__pair__(self.data_type):
                hypo_len = len(sample['type_id']) - sum(sample['type_id'])
                hypothesis_masks[index, :hypo_len] = torch.LongTensor([0] * hypo_len)
                for idy in range(hypo_len, select_len):
                    premise_masks[index, idy] = 0

        if self.__if__pair__(self.data_type):
            batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2,
                'premise_mask': 3,
                'hypothesis_mask': 4
            }
            batch_data = [token_ids, type_ids, masks, premise_masks, hypothesis_masks]
        else:
            batch_info = {
                'token_id': 0,
                'segment_id': 1,
                'mask': 2
            }
            batch_data = [token_ids, type_ids, masks]
        return batch_data, batch_info












