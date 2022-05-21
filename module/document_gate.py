"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-11
"""
import torch
import torch.nn as nn
from module.sub_layers import LayerNorm
from module.multi_head_attn import MultiHeadedAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = LayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class DocumentAttentionLayer(nn.Module):
    def __init__(self, size, dropout=0, head_count=12, hidden_szie=768):
        # dropout暂定为0
        super(DocumentAttentionLayer, self).__init__()
        self.self_attention = MultiHeadedAttention(head_count=head_count, model_dim=size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, hidden_szie, dropout)
        self.layer_norm = LayerNorm(size)

    def forward(self, key, value, query, mask):
        """
        :param key:
        :param value:
        :param query:
        :param mask:
        :return:
        """
        # multi-head attention
        context = self.self_attention(key, value, query, mask)
        # feed-forward layer
        dt = self.feed_forward(context)
        # norm
        out = self.layer_norm(dt)
        return out


class DocumentGate(nn.Module):
    def __init__(self, dim):
        super(DocumentGate, self).__init__()
        # 对cache当中的内容进行attention
        self.document_attention = DocumentAttentionLayer(dim)
        # 上下文的权重
        self.document_weight = nn.Linear(dim, 1, bias=True)
        # 当前句子的权重
        self.sequence_weight = nn.Linear(dim, 1, bias=True)
        # sigmoid_gate
        self.sigmoid_gate = nn.Sigmoid()

    def forward(self, encoder_output,
                document_context, document_state_context, mask):
        """
        :param encoder_output:
        :param document_context:
        :param document_state_context:
        :param mask:
        :return:
        """
        document_output = self.document_attention(
            document_state_context,  # key
            document_context,  # value
            encoder_output,  # query
            mask=mask
        )
        # multi-head attn的输出
        document_output = document_output.squeeze(1) # batch_size * sequence_len * feature_dim
        # 当前sequence对应的context的权重
        weight_d = self.document_weight(document_output)
        # 当前sequence的权重
        weight_s = self.document_weight(encoder_output)
        # gate的权重
        weight_gate = self.sigmoid_gate(weight_d + weight_s)
        # 最终输出
        final_output = weight_gate * encoder_output + (1 - weight_gate) * document_context

        return final_output, weight_gate


class DocumentGateWithOutAttn(nn.Module):
    def __init__(self, dim):
        # 用于蒸馏实验 without multi-head attention
        super(DocumentGateWithOutAttn, self).__init__()
        # 上下文的权重
        self.document_weight = nn.Linear(dim, 1, bias=True)
        # 当前句子的权重
        self.sequence_weight = nn.Linear(dim, 1, bias=True)
        # sigmoid_gate
        self.sigmoid_gate = nn.Sigmoid()

    def forward(self, encoder_output,
                document_context, document_state_context, mask):
        """
        :param encoder_output:
        :param document_context:
        :param document_state_context:
        :param mask:
        :return:
        """
        # 当前sequence对应的context的权重
        weight_d = self.document_weight(document_context)
        # 当前sequence的权重
        weight_s = self.document_weight(encoder_output)
        # gate的权重
        weight_gate = self.sigmoid_gate(weight_d + weight_s)
        # 最终输出
        final_output = weight_gate * encoder_output + (1 - weight_gate) * document_context

        return final_output, weight_gate


if __name__ == '__main__':
    a = DocumentGate(512)
    print('debug use')



















