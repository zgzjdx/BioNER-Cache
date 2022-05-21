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
from module.document_gate import DocumentGate, DocumentGateWithOutAttn
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.use_cache:
            self.document_gate = DocumentGate(hidden_size)
            #self.document_gate = DocumentGateWithOutAttn(hidden_size)
            self.dynamic_cache = DynamicCache(
                cache_dim=opt['rnn_hidden_size'],  # 用于表示lstm的隐层维度
                cache_size=opt['cache_size'],  # cache的大小
                rnn_type=opt['rnn_type'],  # lstm or gru or rnn
                input_size=opt['hidden_size'],  # 用于表示输入的隐层维度
                num_layers=opt['rnn_num_layers'],  # rnn layers
                dropout=opt['rnn_dropout'],  # rnn dropout rate
                opt=opt,
                document_gate=self.document_gate
            )
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
            elif task_type == TaskType.Joint_mrc_ner:
                assert len(label_list) == 2
                out_proj = nn.ModuleList()
                ner_label = label_list[0]
                # for ner
                out_proj.append(nn.Linear(hidden_size, ner_label))
                start_outputs, end_outputs = nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 1)
                # span_embeddings = MultiNonLinearClassifier(hidden_size*2, 1, opt['mrc_dropout'])
                # for mrc
                out_proj.append(start_outputs)
                out_proj.append(end_outputs)
                # out_proj.append(span_embeddings)
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
            with torch.autograd.set_detect_anomaly(True):
                if self.use_cache:
                    # docNER
                    # current_state
                    sequence_output = self.dropout_list[task_id](sequence_output)  # [batch_size, sequence_length, hidden_size]
                    # sequence_output_ = sequence_output.detach()
                    # 对cache进行动态更新
                    document_aware_state, document_flag, cache_logits = self.dynamic_cache(input_ids, sequence_output, attention_mask)
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
        elif task_type == TaskType.Joint_mrc_ner:
            # batch_size * seq_len * feature_dim
            sequence_output = self.dropout_list[task_id](sequence_output)
            batch_size, seq_len, hidden_size = sequence_output.size()
            ner_sequence_output = sequence_output.contiguous().view(-1, sequence_output.size(2))
            ner_logits = self.scoring_list[task_id][0](ner_sequence_output)
            start_logits = self.scoring_list[task_id][1](sequence_output).squeeze(-1)  # [batch, seq_len, 1]
            end_logits = self.scoring_list[task_id][2](sequence_output).squeeze(-1)  # [batch, seq_len, 1]
            # for every position $i$ in sequence, should concate $j$ to
            # predict if $i$ and $j$ are start_pos and end_pos for an entity.
            # [batch, seq_len, seq_len, hidden]
            # start_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
            # # [batch, seq_len, seq_len, hidden]
            # end_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)
            # # [batch, seq_len, seq_len, hidden*2]
            # span_matrix = torch.cat([start_extend, end_extend], 3)
            # # [batch, seq_len, seq_len]
            # span_logits = self.scoring_list[task_id][3](span_matrix).squeeze(-1)
            return ner_logits, start_logits, end_logits
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
