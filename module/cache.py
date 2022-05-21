"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-12
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


def sequence_mask(lengths, max_len=None):
    """
    Create a boolean mask from sequence lengths.
    :param lengths:
    :param max_len:
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1)))


class DynamicCache(nn.Module):
    # 初始化
    def __init__(self, cache_dim, cache_size, rnn_type, input_size, num_layers, dropout, opt):
        super(DynamicCache, self).__init__()
        self.cache_dim = int(cache_dim)
        self.cache_size = int(cache_size)
        # 将encoder输入lstm中, 是否需要用, 待定
        # todo label embedding
        self.cache_embedding_rnn = getattr(nn, rnn_type)(
            input_size=input_size,  # 输入的隐层维度
            hidden_size=cache_dim,  # LSTM的隐层维度
            num_layers=num_layers,  # LSTM层数
            dropout=dropout,
            bidirectional=True,
        )
        # linear和sigmoid函数 用于给当前的状态进行打分
        self.linear = nn.Linear(cache_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
        # 定义一个网络初始的hidden=None
        self.hidden = None
        # key-value cache key存储document状态 value存储encoder的原始状态 score存储sigmoid函数的概率打分
        self.key = []
        self.value = []
        self.score = []
        # 将存入cache的src存储进去, 打印attention用
        self.word_cache = []
        # global attention用, 需要定义额外的document_memory_lengths
        self.document_memory_lengths = None
        self.oldest_index = 0
        self.opt = opt

    def forward(self, input_ids, sequence_output):
        """
        :param input_ids:
        :param sequence_output:
        :return:
        """
        if not self.key:
            # 此时cache为空, 写入, rnn的输出做key, bert的输出做value, sigmoid的输出做score, input_id做id
            rnn_output, _ = self.cache_embedding_rnn(sequence_output, self.hidden)  # batch_size * sequence_len * hidden_size
            cache_logits = torch.sum(self.sigmoid(self.linear(rnn_output)), dim=1) / len(input_ids[-1])  # batch_size * 1
            cache_logits = cache_logits.squeeze()  # batch_size
            assert len(self.key) == len(self.value) == len(self.word_cache) == len(self.score)
            for idx in range(len(rnn_output)):
                # cache还未写满
                if len(self.key) < self.cache_size:
                    self.key.append(rnn_output[idx])
                    self.value.append(sequence_output[idx])
                    self.word_cache.append(list(input_ids[idx]))
                    self.score.append(cache_logits[idx])  # sigmoid = sequence_len * 1
                # cache写满
                else:
                    if self.opt['cache_update_strategy'] == 'normal':
                        # rule1 先进先出
                        if self.oldest_index > self.cache_size:
                            self.oldest_index = 0
                        self.key[self.oldest_index] = rnn_output[idx]
                        self.value[self.oldest_index] = sequence_output[idx]
                        self.word_cache[self.oldest_index] = list(input_ids[idx])
                        self.score[self.oldest_index] = cache_logits[idx]
                        self.oldest_index += 1
                    elif self.opt['cache_update_strategy'] == 'score':
                        # rule2 根据打分函数
                        min_score = min(self.score)
                        min_score_index = self.score.index(min_score)
                        current_score = cache_logits[idx]
                        current_id = list(input_ids[idx])
                        if current_score > min_score:
                            # 相关性高, 保留
                            if current_id in self.word_cache:
                                index_temp = self.word_cache.index(current_id)
                                self.key[index_temp] = rnn_output[idx]
                                self.value[index_temp] = sequence_output[idx]
                                self.score[index_temp] = current_score
                            else:
                                self.key[min_score_index] = rnn_output[idx]
                                self.value[min_score_index] = sequence_output[idx]
                                self.word_cache[min_score_index] = list(input_ids[idx])
                                self.score[min_score_index] = current_score
                        else:
                            pass
                    else:
                        raise ValueError

        else:
            # 若cache不为空
            assert len(self.key) == len(self.value) == len(self.word_cache) == len(self.score)
            rnn_output, _ = self.cache_embedding_rnn(sequence_output,
                                                     self.hidden)  # batch_size * sequence_len * hidden_size
            cache_logits = torch.sum(self.sigmoid(self.linear(rnn_output)), dim=1) / len(input_ids[-1])
            cache_logits = cache_logits.squeeze()
            # cache不为空, 但未写满
            if len(self.key) < self.cache_size:
                diff = self.cache_size - len(self.key)
                for idx in range(len(rnn_output)):
                    # 先写满
                    if idx < diff:
                        self.key.append(rnn_output[idx])
                        self.value.append(sequence_output[idx])
                        self.word_cache.append(list(input_ids[idx]))
                        self.score.append(cache_logits[idx])  # sigmoid = sequence_len * 1
                    # 写满后开始更新
                    else:
                        if self.opt['cache_update_strategy'] == 'normal':
                            # rule1 先进先出
                            if self.oldest_index > self.cache_size:
                                self.oldest_index = 0
                            self.key[self.oldest_index] = rnn_output[idx]
                            self.value[self.oldest_index] = sequence_output[idx]
                            self.word_cache[self.oldest_index] = list(input_ids[idx])
                            self.score[self.oldest_index] = cache_logits[idx]
                            self.oldest_index += 1
                        elif self.opt['cache_update_strategy'] == 'score':
                            # rule2 根据打分函数更新
                            min_score = min(self.score)
                            min_score_index = self.score.index(min_score)
                            current_score = cache_logits[idx]
                            current_id = list(input_ids[idx])
                            if current_score > min_score:
                                # 相关性高, 保留
                                if current_id in self.word_cache:
                                    index_temp = self.word_cache.index(current_id)
                                    self.key[index_temp] = rnn_output[idx]
                                    self.value[index_temp] = sequence_output[idx]
                                    self.score[index_temp] = current_score
                                else:
                                    self.key[min_score_index] = rnn_output[idx]
                                    self.value[min_score_index] = sequence_output[idx]
                                    self.word_cache[min_score_index] = list(input_ids[idx])
                                    self.score[min_score_index] = current_score
                            else:
                                pass
                        else:
                            raise ValueError
            # cache不为空且写满
            else:
                for idx in range(len(rnn_output)):
                    if self.opt['cache_update_strategy'] == 'normal':
                        # rule1 先进先出
                        if self.oldest_index >= self.cache_size:
                            self.oldest_index = 0
                        self.key[self.oldest_index] = rnn_output[idx]
                        self.value[self.oldest_index] = sequence_output[idx]
                        self.word_cache[self.oldest_index] = list(input_ids[idx])
                        self.score[self.oldest_index] = cache_logits[idx]
                        self.oldest_index += 1
                    elif self.opt['cache_update_strategy'] == 'score':
                        # rule2 根据打分函数更新
                        min_score = min(self.score)
                        min_score_index = self.score.index(min_score)
                        current_score = cache_logits[idx]
                        current_id = list(input_ids[idx])
                        if current_score > min_score:
                            # 相关性高, 保留
                            if current_id in self.word_cache:
                                index_temp = self.word_cache.index(current_id)
                                self.key[index_temp] = rnn_output[idx]
                                self.value[index_temp] = sequence_output[idx]
                                self.score[index_temp] = current_score
                            else:
                                self.key[min_score_index] = rnn_output[idx]
                                self.value[min_score_index] = sequence_output[idx]
                                self.word_cache[min_score_index] = list(input_ids[idx])
                                self.score[min_score_index] = current_score
                        else:
                            pass
                    else:
                        raise ValueError

        return cache_logits


if __name__ == '__main__':
    a = torch.randn(1, 2, 3, 4, 5)
    print(a.shape)
    # Returns the total number of elements in the input tensor.
    b = a.numel()
    print(b)
    print(sequence_mask(a))