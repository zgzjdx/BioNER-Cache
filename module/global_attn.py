"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-17
"""
import torch
import torch.nn as nn
from module.sub_layers import aeq
from module.cache import sequence_mask


class GlobalAttention(nn.Module):
    def __init__(self, dim, coverage=False, attn_type='dot'):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ['dot', 'general', 'mlp']), 'please select a valid attention type'

        if self.attn_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.liner_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == 'mlp'
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)
        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        :param h_t:
        :param h_s:
        :return:
        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(src_len, tgt_len)

        if self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t_ = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # 处理文本计算attention权重的时候，很容易得到的权重矩阵shape是 [batch_size, sequence_length]，
            # 然后需要相乘的隐状态矩阵是 [batch_size, sequence_length, hidden_size]。
            # 按照attention的计算方式，实际上就是权重矩阵中每一行的数值分别乘以隐状态矩阵中每一行的对应位置的隐状态，
            # 这个过程当然可以写循环，也可以简单的使用bmm函数计算，
            # 先将权重矩阵reshape成 [batch_size, 1, sequence_length]然后bmm(weigths_matrix, hidden_matrix)
            # 然后得到的结果就是attention计算的结果了。
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)
            # Tensor.contiguous()函数不会对原始数据进行任何修改，而仅仅对其进行复制，并在内存空间上进行对齐，
            # 即在内存空间上，tensor元素的内存地址保持连续。 注意 transpose(), narrow(), expand()函数
            uh = self.liner_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None):
        """
        :param input:
        :param memory_bank:
        :param memory_lengths:
        :param coverage:
        :return:
        """
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch_, sourceL_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)

        align = self.score(input, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1) # Make it broadcastable
            # inf正无穷, -inf负无穷
            # 如果你希望正确的判断 Inf 和 Nan 值，那么你应该使用 math 模块的 math.isinf 和 math.isnan 函数
            align.data.masked_fill_(1-mask, -float('inf'))

        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        c = torch.bmm(align_vectors, memory_bank)
        concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ['general', 'dot']:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        return attn_h, align_vectors


if __name__ == '__main__':
    pass