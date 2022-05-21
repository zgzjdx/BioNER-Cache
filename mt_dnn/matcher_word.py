# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import torch
import torch.nn as nn
import json
from torch.autograd import Variable
from pytorch_pretrained_bert.modeling import BertConfig, BertLayerNorm, BertModel
from mt_dnn.mt_dnn_utils import MultiNonLinearClassifier
from module.dropout_wrapper import DropoutWrapper
from module.san import SANClassifier
from data_utils.task_def import EncoderModelType, TaskType
from module.cache import DynamicCache
from module.document_gate import DocumentGate
from module.sub_layers import LayerNorm
from module.crf import CRF
from module.pooling import max_pooling, mean_pooling, mean_pooling_sqrt
from module.self_attention import SelfAttention


class LinearPooler(nn.Module):
    def __init__(self, hidden_size):
        super(LinearPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SANBertNetwork(nn.Module):
    def __init__(self, opt, bert_config=None):
        super(SANBertNetwork, self).__init__()  # 先找到父类nn.Moudle, 将其对象转化为SANBertNetwork的对象
        self.dropout_list = nn.ModuleList()
        self.encoder_type = opt['encoder_type']
        if opt['encoder_type'] == EncoderModelType.ROBERTA:
            from fairseq.models.roberta import RobertaModel
            self.bert = RobertaModel.from_pretrained(opt['init_checkpoint'])
            hidden_size = self.bert.args.encoder_embed_dim
            self.pooler = LinearPooler(hidden_size)
        # 若要支持其它的预训练模型,在这里添加
        else: 
            self.bert_config = BertConfig.from_dict(opt)
            # print(self.bert_config)
            self.bert = BertModel(self.bert_config)
            hidden_size = self.bert_config.hidden_size

        if opt.get('dump_feature', False):
            self.opt = opt
            return
        if opt['update_bert_opt'] > 0:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.decoder_opt = opt['answer_opt']
        self.task_types = opt["task_types"]
        self.use_cache = opt['use_cache']
        # self.use_luo = opt['use_luo']
        # if self.use_luo:
        #     self.bigru = nn.GRU(input_size=opt['hidden_size'], hidden_size=opt['hidden_size']//2, bidirectional=True)
        #     self.selfattn = SelfAttention(opt['hidden_size'])
        if self.use_cache:
            self.dynamic_cache = DynamicCache(
                cache_dim=opt['rnn_hidden_size'],  # 用于表示lstm的隐层维度
                cache_size=opt['cache_size'],  # cache的大小
                rnn_type=opt['rnn_type'],  # lstm or gru or rnn
                input_size=opt['hidden_size'],  # 用于表示输入的隐层维度
                num_layers=opt['rnn_num_layers'],  # rnn layers
                dropout=opt['rnn_dropout'],  # rnn dropout rate
                opt=opt,
            )
            self.document_gate = DocumentGate(hidden_size)
            self.layer_norm = LayerNorm(hidden_size)
        # todo 这样的docL_NER是不支持多个数据集的, 未来要支持多个数据集
        self.use_crf = opt['use_crf']
        self.scoring_list = nn.ModuleList()
        labels = [ls for ls in opt['label_size'].split(';')]
        task_dropout_p = opt['tasks_dropout_p']

        for task, lab in enumerate(labels):
            try:
                lab = int(lab)
            except:
                label_list = json.loads(lab)
            decoder_opt = self.decoder_opt[task]
            task_type = self.task_types[task]
            dropout = DropoutWrapper(task_dropout_p[task], opt['vb_dropout'])
            self.dropout_list.append(dropout)
            if task_type == TaskType.Span:
                assert decoder_opt != 1
                out_proj = nn.Linear(hidden_size, 2)
            elif task_type == TaskType.SequenceLabeling:
                out_proj = nn.ModuleList()
                if self.use_crf:
                    crf_liner = nn.Linear(opt['hidden_size'], int(lab))
                    crf = CRF(num_tags=int(lab), batch_first=True)
                    out_proj.append(crf_liner)
                    out_proj.append(crf)
                else:
                    out_proj.append(nn.Linear(hidden_size, lab))
            elif task_type == TaskType.ReadingComprehension:
                start_outputs, end_outputs, span_embeddings = nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 1),\
                                                        MultiNonLinearClassifier(hidden_size*2, 1, opt['mrc_dropout'])
                out_proj = nn.ModuleList()
                out_proj.append(start_outputs)
                out_proj.append(end_outputs)
                out_proj.append(span_embeddings)
            elif task_type == TaskType.Joint:
                out_proj = nn.ModuleList()
                # todo 用字典好一些
                assert len(label_list) == 2
                ner_label = label_list[0]
                cls_label = label_list[-1]
                # for ner
                out_proj.append(nn.Linear(hidden_size, ner_label))
                # for cls
                out_proj.append(nn.Linear(hidden_size, cls_label))
            elif task_type == TaskType.Joint_mt:
                out_proj = nn.ModuleList()
                assert len(label_list) == 2
                ner_label = label_list[0]
                mt_label = label_list[-1]
                # for ner
                out_proj.append(nn.Linear(hidden_size, ner_label))
                # for mt
                out_proj.append(nn.Linear(hidden_size, mt_label))
            elif task_type == TaskType.Joint_mt_three:
                out_proj = nn.ModuleList()
                assert len(label_list) == 3
                ner_label = label_list[0]
                mt_label = label_list[1]
                cls_label = label_list[2]
                # for ner
                out_proj.append(nn.Linear(hidden_size, ner_label))
                # for mt
                out_proj.append(nn.Linear(hidden_size, mt_label))
                # for cls
                out_proj.append(nn.Linear(hidden_size, cls_label))
            elif task_type == TaskType.Joint_cls_three:
                out_proj = nn.ModuleList()
                assert len(label_list) == 3
                ner_label = label_list[0]
                bcls_label = label_list[1]
                mcls_label = label_list[2]
                # for ner
                out_proj.append(nn.Linear(hidden_size, ner_label))
                # for bcls
                out_proj.append(nn.Linear(hidden_size, bcls_label))
                # for mcls
                out_proj.append(nn.Linear(hidden_size, mcls_label))
            elif task_type == TaskType.Joint_all:
                out_proj = nn.ModuleList()
                assert len(label_list) == 4
                ner_label = label_list[0]
                mt_label = label_list[1]
                bcls_label = label_list[2]
                mcls_label = label_list[3]
                # for ner
                out_proj.append(nn.Linear(hidden_size, ner_label))
                # for mt
                out_proj.append(nn.Linear(hidden_size, mt_label))
                # for bcls
                out_proj.append(nn.Linear(hidden_size, bcls_label))
                # for mcls
                out_proj.append(nn.Linear(hidden_size, mcls_label))
            elif task_type == TaskType.Joint_bCLS_mCLS:
                out_proj = nn.ModuleList()
                assert len(label_list) == 2
                bcls_label = label_list[0]
                mcls_label = label_list[1]
                # for bcls
                out_proj.append(nn.Linear(hidden_size, bcls_label))
                # for mcls
                out_proj.append(nn.Linear(hidden_size, mcls_label))
            else:
                if decoder_opt == 1:
                    out_proj = SANClassifier(hidden_size, hidden_size, lab, opt, prefix='answer', dropout=dropout)
                else:
                    out_proj = nn.Linear(hidden_size, lab)
            self.scoring_list.append(out_proj)

        self.opt = opt
        self._my_init()

    def _my_init(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02 * self.opt['init_ratio'])
            elif isinstance(module, BertLayerNorm):
                # Slightly different from the BERT pytorch version, which should be a bug.
                # Note that it only affects on training from scratch. For detailed discussions, please contact xiaodl@.
                # Layer normalization (https://arxiv.org/abs/1607.06450)
                # support both old/latest version
                if 'beta' in dir(module) and 'gamma' in dir(module):
                    module.beta.data.zero_()
                    module.gamma.data.fill_(1.0)
                else:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def query_and_read(self, input_ids, sequence_output, document_context, attention_mask):
        """
        :param input_ids: batch_size * seq_len
        :param sequence_output: batch_size * seq_len * feature_dim
        :param document_context: cache_size * feature_dim
        :param attention_mask: batch_size * seq_len
        :return:
        """
        for idx in range(len(self.dynamic_cache.word_cache)):
            if idx == 0:
                cache_word_ids = self.dynamic_cache.word_cache[idx].data.unsqueeze(0)
            else:
                # cache_size * 1
                cache_word_ids = torch.cat(
                    (cache_word_ids, self.dynamic_cache.word_cache[idx].data.unsqueeze(0)), 0)

        batch_size, seq_len, _ = sequence_output.size()
        # 占位
        history_state = sequence_output
        for idx in range(batch_size):
            for idy in range(seq_len):
                # current_state = sequence_output[idx][idy]
                current_word_id = input_ids[idx][idy]
                if current_word_id in cache_word_ids:
                    index = self.dynamic_cache.word_cache.index(current_word_id)
                    history_state[idx][idy] = document_context[index]
                else:
                    continue
        padding_idx = 0
        mask = input_ids.eq(padding_idx).unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, feature_dim]
        # print(mask.shape)
        return history_state, mask

    def compute_global_representation(self, sequence_output, input_ids, attention_mask):
        """
        :param sequence_output:
        :param input_ids:
        :param attention_mask:
        :return:
        """
        document_context = None  # value
        document_state_context = None  # key
        document_flag = False
        context_aware_state = None  # final_output
        try:
            # cache为空时, 用异常处理
            if not self.dynamic_cache.key:
                pass
            elif sequence_output.size(0) != self.opt['batch_size']:
                pass
            else:
                # 因为value=key 构造temp_tensor防止梯度回传两次
                assert len(self.dynamic_cache.value) == len(self.dynamic_cache.key) == \
                       len(self.dynamic_cache.score) == len(self.dynamic_cache.word_cache)

                for idx in range(len(self.dynamic_cache.value)):
                    # document_index = reorder_index[idx]
                    # 参考attention is all you need中 value=key
                    temp_value = Variable(self.dynamic_cache.value[idx].data, requires_grad=True)
                    temp_key = Variable(self.dynamic_cache.value[idx].data, requires_grad=True)
                    if idx == 0:
                        document_context = temp_value.unsqueeze(0)
                        document_state_context = temp_key.unsqueeze(0)
                    else:
                        # cache_size * hidden_size
                        document_context = torch.cat((document_context, temp_value.unsqueeze(0)), 0)
                        # cache_size * hidden_size
                        document_state_context = torch.cat((document_state_context, temp_key.unsqueeze(0)), 0)
                document_flag = True
        except:
            pass
        # make mask, 用于辅助multi-head attn的计算
        if document_flag:
            # todo 这里MEAN和MAX有些问题, 特征值都变小了
            # todo normal_cache_normal_cat无法训练, 损失不变
            if self.opt['query_and_read_strategy'] == 'DNN':
                # rule4 把context_cache的存储内容压缩成一个向量, 然后进入context_gate
                # 这里其实还有更多的组合
                input_dim = document_context.size(0)
                out_dim = sequence_output.size(0)
                # todo 这里是不对的
                project_layer = nn.Linear(input_dim, out_dim).cuda()
                history_state = project_layer(document_context.transpose(0, 2))
                history_state = history_state.transpose(0, 2)
                context_aware_state, weight = self.document_gate(
                    encoder_output=sequence_output,  # current_state, query [batch_size, seq_len, dim]
                    document_context=history_state,  # history_state, value [batch_size, value_len, dim]
                    document_state_context=history_state,  # key=value=history_state [batch_size, value_len, dim]
                    mask=attention_mask
                )
            else:
                history_state, mask = self.query_and_read(input_ids, sequence_output, document_context, attention_mask)
                if self.opt['query_and_read_strategy'] == 'normal_add':
                    # rule1 直接找最相似的上下文, 然后做element-wise add和平均, mask用于辅助rule2的multi-head attention计算
                    context_aware_state = torch.add(sequence_output, history_state).div(2)
                elif self.opt['query_and_read_strategy'] == 'normal_cat':
                    # rule2 直接找最相似的上下文, 然后拼接做维度转换, mask用于辅助rule2的multi-head attention计算
                    context_aware_state = torch.cat((sequence_output, history_state), dim=0).cuda()
                    dim_convert = torch.nn.Linear(context_aware_state.size(0), sequence_output.size(0), bias=False).cuda()
                    context_aware_state = dim_convert(context_aware_state.transpose(0, -1))
                    context_aware_state = context_aware_state.transpose(0, -1)
                elif self.opt['query_and_read_strategy'] == 'context_gate':
                    # rule3 直接找最相似的上下文, 然后用multi-head attention和context gate, 利用context_gate计算final_state
                    # context_aware_state batch_size * seq_len * feature_size
                    context_aware_state, weight = self.document_gate(
                        encoder_output=sequence_output,  # current_state, query [batch_size, seq_len, dim]
                        document_context=history_state,  # history_state, value [batch_size, value_len, dim]
                        document_state_context=history_state,  # key=value=history_state [batch_size, value_len, dim]
                        mask=mask
                    )
                else:
                    raise ValueError

        return context_aware_state, document_flag

    def _get_model_outputs(self, key):
        """
        for task embedding compute
        :param key:
        :return:
        """
        if key == 'multihead_output':
            # get list (layers) of multi-head module outputs
            return [layer.attention.self.multihead_output for layer in self.bert_encoder.layer]
        elif key == 'layer_output':
            # get list of encoder LayerNorm Layer outputs
            return [layer.outpu.layer_output for layer in self.bert_encoder.layer]
        elif key == 'layer_output':
            # get the final output of the model
            return self.bert_pooler.cls_output
        else:
            raise ValueError("Key not found: %s" %(key))

    def forward(self, input_ids, token_type_ids, attention_mask, premise_mask=None, hyp_mask=None, task_id=0):
        """
        todo 添加参数的维度注释
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param premise_mask:
        :param hyp_mask:
        :param task_id:
        :return:
        """
        if self.encoder_type == EncoderModelType.ROBERTA:
            sequence_output = self.bert.extract_features(input_ids)
            pooled_output = self.pooler(sequence_output)
        else:
            # all_encoder_layers, list, 12层bert的输出
            # Outputs: Tuple of (encoded_layers, pooled_output)
            # `encoded_layers`: controled by `output_all_encoded_layers` argument:
            #      - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
            #      of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
            #      encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            #      - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
            #      to the last attention block of shape [batch_size, sequence_length, hidden_size],
            #      `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            #       classifier pretrained on top of the hidden state associated to the first character of the
            #       input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
            all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]

        decoder_opt = self.decoder_opt[task_id]
        task_type = self.task_types[task_id]
        if task_type == TaskType.ReadingComprehension:
            assert decoder_opt != 1
            # todo 增加维度的注释, 下同
            sequence_output = self.dropout_list[task_id](sequence_output)  # [batch_size, seq_len, feature_dim]
            batch_size, seq_len, hid_size = sequence_output.size()
            start_scores = self.scoring_list[task_id][0](sequence_output).squeeze(-1)
            end_scores = self.scoring_list[task_id][1](sequence_output).squeeze(-1)
            start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
            end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
            span_matrix = torch.cat([start_extend, end_extend], 3)
            span_logits = self.scoring_list[task_id][-1](span_matrix).squeeze(-1)
            return start_scores, end_scores, span_logits
        elif task_type == TaskType.SequenceLabeling:
            if self.use_cache:
                # docNER
                # current_state
                sequence_output = self.dropout_list[task_id](sequence_output)  # [batch_size, sequence_length, hidden_size]
                # 取cache的value用multi-head和context-gate来计算上下文结果
                document_aware_state, document_flag = self.compute_global_representation(sequence_output,
                                                                                         input_ids, attention_mask)

                # 对cache进行动态更新
                cache_logits = self.dynamic_cache(input_ids, sequence_output)
                # 计算logits
                if document_flag:
                    # cache不为空的情况
                    sequence_output = self.dropout_list[task_id](document_aware_state)
                else:
                    # cache为空的情况, 即为普通的ner
                    sequence_output = sequence_output
                if self.use_crf:
                    # 使用crf
                    logits = self.crf_liner(sequence_output)
                else:
                    # 不使用crf
                    sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))
                    # nn.linear
                    logits = self.scoring_list[task_id][0](sequence_output)
                return logits, cache_logits
            else:
                # 不使用cache
                if self.use_crf:
                    # 使用crf
                    # todo crf learning_rate
                    sequence_output = self.dropout_list[task_id](sequence_output)  # [batch_size, seq_len, feature_dim]
                    logits = self.scoring_list[task_id][0](sequence_output)  # [batch_size, seq_len, num_labels]
                    # logits = self.crf_liner(sequence_output)
                    return logits, self.scoring_list[task_id][1]
                else:
                    # 不使用crf
                    sequence_output = self.dropout_list[task_id](sequence_output)  # [batch_size, seq_len, feature_dim]
                    sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))  # [batch_size * seq_len, feature_dim]
                    # nn.linear
                    logits = self.scoring_list[task_id][-1](sequence_output)
                    return logits
        elif task_type == TaskType.Joint:
            sequence_output = self.dropout_list[task_id](sequence_output)
            if self.opt['cls_pooling'] == 'CLS':
                # 用[cls]做分类
                pooled_output = self.dropout_list[task_id](pooled_output)  # [batch_size, feature_dim]
            elif self.opt['cls_pooling'] == 'MAX':
                # 用max_pooling做分类
                token_embeddings = sequence_output # [batch_size, seq_len, feature_dim]
                pooled_output = max_pooling(token_embeddings, attention_mask)  # [batch_size, feature_dim]
            elif self.opt['cls_pooling'] == 'MEAN':
                # 用mean_pooling做分类
                token_embeddings = sequence_output  # [batch_size, seq_len, feature_dim]
                pooled_output = mean_pooling(token_embeddings, attention_mask)  # [batch_size, feature_dim]
            elif self.opt['cls_pooling'] == 'MEAN_sqrt':
                # 用mean_pooling做分类加个normalization
                token_embeddings = sequence_output  # [batch_size, seq_len, feature_dim]
                pooled_output = mean_pooling_sqrt(token_embeddings, attention_mask)  # [batch_size, feature_dim]
            else:
                raise ValueError
              #[batch_size* seq_len, feature_dim]
            sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))  # [batch_size * seq_len, feature_dim]
            ner_logits = self.scoring_list[task_id][0](sequence_output)  # [batch_size * seq_len, label_num]
            cls_logits = self.scoring_list[task_id][-1](pooled_output)   # [batch_size * seq_len, label_num]
            return ner_logits, cls_logits
        elif task_type == TaskType.Joint_mt:
            sequence_output = self.dropout_list[task_id](sequence_output)
            sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))
            ner_logits = self.scoring_list[task_id][0](sequence_output)
            mt_logits = self.scoring_list[task_id][-1](sequence_output)
            return ner_logits, mt_logits
        elif task_type == TaskType.Joint_mt_three:
            sequence_output = self.dropout_list[task_id](sequence_output)
            pooled_output = self.dropout_list[task_id](pooled_output)
            sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))
            ner_logits = self.scoring_list[task_id][0](sequence_output)
            mt_logits = self.scoring_list[task_id][1](sequence_output)
            cls_logits = self.scoring_list[task_id][2](pooled_output)
            return ner_logits, mt_logits, cls_logits
        elif task_type == TaskType.Joint_cls_three:
            sequence_output = self.dropout_list[task_id](sequence_output)
            sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))
            pooled_output = self.dropout_list[task_id](pooled_output)
            ner_logits = self.scoring_list[task_id][0](sequence_output)
            bcls_logits = self.scoring_list[task_id][1](pooled_output)
            mcls_logits = self.scoring_list[task_id][2](pooled_output)
            return ner_logits, bcls_logits, mcls_logits
        elif task_type == TaskType.Joint_all:
            sequence_output = self.dropout_list[task_id](sequence_output)
            if self.opt['cls_pooling'] == 'CLS':
                # 用[cls]做分类
                pooled_output = self.dropout_list[task_id](pooled_output)  # [batch_size, feature_dim]
            elif self.opt['cls_pooling'] == 'MAX':
                token_embeddings = sequence_output # [batch_size, seq_len, feature_dim]
                pooled_output = max_pooling(token_embeddings, attention_mask)  # [batch_size, feature_dim]
            elif self.opt['cls_pooling'] == 'MEAN':
                token_embeddings = sequence_output  # [batch_size, seq_len, feature_dim]
                pooled_output = mean_pooling(token_embeddings, attention_mask)  # [batch_size, feature_dim]
            elif self.opt['cls_pooling'] == 'MEAN_sqrt':
                token_embeddings = sequence_output  # [batch_size, seq_len, feature_dim]
                pooled_output = mean_pooling_sqrt(token_embeddings, attention_mask)  # [batch_size, feature_dim]
            else:
                raise ValueError
            sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))
            ner_logits = self.scoring_list[task_id][0](sequence_output)
            mt_logits = self.scoring_list[task_id][1](sequence_output)
            bcls_logits = self.scoring_list[task_id][2](pooled_output)
            mcls_logits = self.scoring_list[task_id][3](pooled_output)
            return ner_logits, mt_logits, bcls_logits, mcls_logits
        elif task_type == TaskType.Joint_bCLS_mCLS:
            pooled_output = self.dropout_list[task_id](pooled_output)
            bcls_logits = self.scoring_list[task_id][0](pooled_output)
            mcls_logits = self.scoring_list[task_id][1](pooled_output)
            return bcls_logits, mcls_logits
        else:
            if decoder_opt == 1:
                max_query = hyp_mask.size(1)
                assert max_query > 0
                assert premise_mask is not None
                assert hyp_mask is not None
                hyp_mem = sequence_output[:, :max_query, :]
                logits = self.scoring_list[task_id](sequence_output, hyp_mem, premise_mask, hyp_mask)
            else:
                if self.opt['cls_pooling'] == 'CLS':
                    # 用[cls]做分类
                    pooled_output = self.dropout_list[task_id](pooled_output)  # [batch_size, feature_dim]
                elif self.opt['cls_pooling'] == 'MAX':
                    # 用max_pooling做分类
                    token_embeddings = self.dropout_list[task_id](sequence_output)  # [batch_size, seq_len, feature_dim]
                    # # for Luo
                    # token_embeddings = self.bigru(token_embeddings)
                    # token_embeddings = self.selfattn(token_embeddings)
                    # pooled_output = max_pooling(token_embeddings, attention_mask)  # [batch_size, feature_dim]
                elif self.opt['cls_pooling'] == 'MEAN':
                    # 用mean_pooling做分类
                    token_embeddings = self.dropout_list[task_id](sequence_output)  # [batch_size, seq_len, feature_dim]
                    pooled_output = mean_pooling(token_embeddings, attention_mask)  # [batch_size, feature_dim]
                elif self.opt['cls_pooling'] == 'MEAN_sqrt':
                    # 用mean_pooling做分类加个normalization
                    token_embeddings = self.dropout_list[task_id](sequence_output)  # [batch_size, seq_len, feature_dim]
                    pooled_output = mean_pooling_sqrt(token_embeddings, attention_mask)  # [batch_size, feature_dim]
                else:
                    raise ValueError
                logits = self.scoring_list[task_id](pooled_output)

            return logits
