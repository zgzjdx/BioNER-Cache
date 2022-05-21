"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-11
"""
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from module.sub_layers import aeq


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        """
        :param head_count:
        :param model_dim:
        :param dropout:
        """
        # 768 / 8 = 96
        assert model_dim % head_count == 0
        # 向下取整
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        # 维度不会发生变化
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        """
        :param key: [batch_size, key_length, dim]
        :param value: [batch_size, key_length, dim]
        :param query: [batch_size, query_length, dim]
        :param mask: binary mask indicating which keys have non-zero attention [batch_size, query_len, key_len]
        :return:
        """
        batch, key_len, dim = key.size()
        batch_, value_len, dim_ = value.size()
        aeq(batch, batch_)
        aeq(key_len, value_len)
        aeq(dim, dim_)
        batch_, query_len, dim_ = query.size()
        aeq(batch, batch_)
        aeq(key_len, query_len)
        aeq(dim, dim_)
        if mask is not None:
            batch_, query_len_, key_len_ = mask.size()
            aeq(batch, batch_)
            aeq(query_len == query_len_)
            aeq(key_len, key_len_)

        batch_size = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """
            :param x:
            :return:
            """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """
            :param x:
            :return:
            """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # project key, value and query.
        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))

        # calculate and scala scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e18)
        # mask必须是一个 ByteTensor 而且shape必须和 a一样 并且元素只能是 0或者1 ，
        # 是将 mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value

        # apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)

        batch_, q_len_, d_ = output.size()
        aeq(batch, batch_)
        aeq(query_len, q_len_)
        aeq(dim, d_)
        return output



















