"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-21
"""
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from module.optimization import BertAdam as Adam
from module.bert_optim import Adamax, RAdam
from module.my_optim import EMA
from mt_dnn.matcher import SANBertNetwork
from data_utils.task_def import TaskType
from data_utils.utils import AverageMeter
from mt_dnn.dice_loss import mrc_loss_compute

logger = logging.getLogger(__name__)


class MTDnnModel(object):
    def __init__(self, opt, state_dict=None, num_train_step=1):
        self.config = opt
        self.updates = state_dict['updates'] if state_dict and 'updates' in state_dict else 0
        self.local_updates = 0
        self.train_loss = AverageMeter()
        self.network = SANBertNetwork(opt)
        
        if state_dict and 'scibert' not in opt['init_checkpoint'] and 'pubmedbert' not in opt['init_checkpoint']:
            self.network.load_state_dict(state_dict['state'], strict=False)
        else:
            self.network.load_state_dict(state_dict, strict=False)
        #if state_dict:
            #self.network.load_state_dict(state_dict['state'], strict=False)
        # 用于单机多卡训练
        self.mnetwork = nn.DataParallel(self.network) if opt['multi_gpu_on'] else self.network
        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        self.num_examples = 0
        self.total_num_examples = 0
        self.global_feature_dict = {}
        if opt['cuda']:
            self.network.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_parameters = [
            {'params': [p for n, p in self.network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # note that adamax are modified based on the BERT code
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, opt['learning_rate'],
                                       weight_decay=opt['weight_decay'])

        elif opt['optimizer'] == 'adamax':
            self.optimizer = Adamax(optimizer_parameters,
                                    opt['learning_rate'],
                                    warmup=opt['warmup'],
                                    t_total=num_train_step,
                                    max_grad_norm=opt['grad_clipping'],
                                    schedule=opt['warmup_schedule'],
                                    weight_decay=opt['weight_decay'])
            if opt.get('have_lr_scheduler', False): opt['have_lr_scheduler'] = False
        elif opt['optimizer'] == 'radam':
            self.optimizer = RAdam(optimizer_parameters,
                                   opt['learning_rate'],
                                   warmup=opt['warmup'],
                                   t_total=num_train_step,
                                   max_grad_norm=opt['grad_clipping'],
                                   schedule=opt['warmup_schedule'],
                                   eps=opt['adam_eps'],
                                   weight_decay=opt['weight_decay'])
            if opt.get('have_lr_scheduler', False): opt['have_lr_scheduler'] = False
            # The current radam does not support FP16.
            opt['fp16'] = False
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(optimizer_parameters,
                                            opt['learning_rate'],
                                            rho=0.95)
        elif opt['optimizer'] == 'adam':
            self.optimizer = Adam(optimizer_parameters,
                                  lr=opt['learning_rate'],
                                  warmup=opt['warmup'],
                                  t_total=num_train_step,
                                  max_grad_norm=opt['grad_clipping'],
                                  schedule=opt['warmup_schedule'],
                                  weight_decay=opt['weight_decay'])
            if opt.get('have_lr_scheduler', False): opt['have_lr_scheduler'] = False
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if opt['fp16']:
            try:
                from apex import amp
                global amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.network, self.optimizer, opt_level=opt['fp16_opt_level'])
            self.network = model
            self.optimizer = optimizer

        if opt.get('have_lr_scheduler', False):
            if opt.get('scheduler_type', 'rop') == 'rop':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=opt['lr_gamma'], patience=3)
            elif opt.get('scheduler_type', 'rop') == 'exp':
                self.scheduler = ExponentialLR(self.optimizer, gamma=opt.get('lr_gamma', 0.95))
            else:
                milestones = [int(step) for step in opt.get('multi_step_lr', '10,20,30').split(',')]
                self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=opt.get('lr_gamma'))
        else:
            self.scheduler = None

        self.ema = None
        if opt['ema_opt'] > 0:
            self.ema = EMA(self.config['ema_gamma'], self.network)
            if opt['cuda']:
                self.ema.cuda()

        self.para_swapped = False
        # zero optimizer grad
        self.optimizer.zero_grad()

    def setup_ema(self):
        if self.config['ema_opt']:
            self.ema.setup()

    def update_ema(self):
        if self.config['ema_opt']:
            self.ema.update()

    def eval(self):
        if self.config['ema_opt']:
            self.ema.swap_parameters()
            self.para_swapped = True

    def train(self):
        if self.para_swapped:
            self.ema.swap_parameters()
            self.para_swapped = False

    def update(self, batch_meta, batch_data):
        """
        :param batch_meta:
        :param batch_data:
        :return:
        """
        self.network.train()  # Sets the module in the training mode.
        # todo 简化代码和命名规范化
        task_type = batch_meta['task_type']
        if task_type == TaskType.Joint:
            ner_label = batch_data[batch_meta['ner_label']]
            cls_label = batch_data[batch_meta['cls_label']]
        elif task_type == TaskType.Joint_mt:
            ner_label = batch_data[batch_meta['ner_label']]
            mt_label = batch_data[batch_meta['mt_label']]
        elif task_type == TaskType.Joint_mt_three:
            ner_label = batch_data[batch_meta['ner_label']]
            mt_label = batch_data[batch_meta['mt_label']]
            cls_label = batch_data[batch_meta['cls_label']]
        elif task_type == TaskType.Joint_cls_three:
            ner_label = batch_data[batch_meta['ner_label']]
            bcls_label = batch_data[batch_meta['bcls_label']]
            mcls_label = batch_data[batch_meta['mcls_label']]
        elif task_type == TaskType.Joint_all:
            ner_label = batch_data[batch_meta['ner_label']]
            mt_label = batch_data[batch_meta['mt_label']]
            bcls_label = batch_data[batch_meta['bcls_label']]
            mcls_label = batch_data[batch_meta['mcls_label']]
        elif task_type == TaskType.Joint_bCLS_mCLS:
            bcls_label = batch_data[batch_meta['bcls_label']]
            mcls_label = batch_data[batch_meta['mcls_label']]
        else:
            labels = batch_data[batch_meta['label']]
        soft_labels = None
        if self.config.get('mkd_opt', 0) > 0 and ('soft_label' in batch_meta):
            soft_labels = batch_meta['soft_label']

        if task_type == TaskType.ReadingComprehension:
             # If impossible=True, it denotes the (query, context) pair does not have answers
             # (the context does not have entities with the entity type C) .
             # On the contrary, it denotes the (query, context) has at least one answer
             # (the context has at least one entity with the entity type C).
            start_labels_index = batch_meta['start_labels']
            start_labels = batch_data[start_labels_index]
            end_labels_index = batch_meta['end_labels']
            end_labels = batch_data[end_labels_index]
            start_label_mask_index = batch_meta['start_label_mask']
            start_label_mask = batch_data[start_label_mask_index]
            end_label_mask_index = batch_meta['end_label_mask']
            end_label_mask = batch_data[end_label_mask_index]
            match_labels_index = batch_meta['match_labels']
            match_labels = batch_data[match_labels_index]

            if self.config["cuda"]:
                start_labels = start_labels.cuda(non_blocking=True)
                end_labels = end_labels.cuda(non_blocking=True)
                start_label_mask = start_label_mask.cuda(non_blocking=True)
                end_label_mask = end_label_mask.cuda(non_blocking=True)
                match_labels = match_labels.cuda(non_blocking=True)
            start_labels.requires_grad = False
            end_labels.requires_grad = False
            start_label_mask.requires_grad = False
            end_label_mask.requires_grad = False
            match_labels.requires_grad = False
        elif task_type == TaskType.Joint:
            y_ner = ner_label
            y_cls = cls_label
            if self.config['cuda']:
                y_ner = y_ner.cuda(non_blocking=True)
                y_cls = y_cls.cuda(non_blocking=True)
            y_ner.requires_grad = False
            y_cls.requires_grad = False
        elif task_type == TaskType.Joint_mt:
            y_ner = ner_label
            y_mt = mt_label
            if self.config['cuda']:
                y_ner = y_ner.cuda(non_blocking=True)
                y_mt = y_mt.cuda(non_blocking=True)
            y_ner.requires_grad = False
            y_mt.requires_grad = False
        elif task_type == TaskType.Joint_mt_three:
            y_ner = ner_label
            y_mt = mt_label
            y_cls = cls_label
            if self.config['cuda']:
                y_ner = y_ner.cuda(non_blocking=True)
                y_mt = y_mt.cuda(non_blocking=True)
                y_cls = y_cls.cuda(non_blocking=True)
            y_ner.requires_grad = False
            y_mt.requires_grad = False
            y_cls.requires_grad = False
        elif task_type == TaskType.Joint_cls_three:
            y_ner = ner_label
            y_bcls = bcls_label
            y_mcls = mcls_label
            if self.config['cuda']:
                y_ner = y_ner.cuda(non_blocking=True)
                y_bcls = y_bcls.cuda(non_blocking=True)
                y_mcls = y_mcls.cuda(non_blocking=True)
        elif task_type == TaskType.Joint_all:
            y_ner = ner_label
            y_mt = mt_label
            y_bcls = bcls_label
            y_mcls = mcls_label
            if self.config['cuda']:
                y_ner = y_ner.cuda(non_blocking=True)
                y_mt = y_mt.cuda(non_blocking=True)
                y_bcls = y_bcls.cuda(non_blocking=True)
                y_mcls = y_mcls.cuda(non_blocking=True)
        elif task_type == TaskType.Joint_bCLS_mCLS:
            y_bcls = bcls_label
            y_mcls = mcls_label
            if self.config['cuda']:
                y_bcls = y_bcls.cuda(non_blocking=True)
                y_mcls = y_mcls.cuda(non_blocking=True)
        else:
            y = labels
            if task_type == TaskType.Ranking:
                y = y.contiguous().view(-1, batch_meta['pairwise_size'])[:, 0]
            if self.config['cuda']:
                y = y.cuda(non_blocking=True)
            y.requires_grad = False

        task_id = batch_meta['task_id']
        inputs = batch_data[:batch_meta['input_len']]  # 取前3个,即除了label以外的
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)

        if self.config.get('weighted_on', False):
            if self.config['cuda']:
                weight = batch_data[batch_meta['factor']].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta['factor']]

        if task_type == TaskType.Joint:
            # cls_loss and ner_loss compute
            ner_logits, cls_logits = self.mnetwork(*inputs)
            y_ner = y_ner.view(-1)
            ner_loss = F.cross_entropy(ner_logits, y_ner, ignore_index=-1)
            cls_loss = F.cross_entropy(cls_logits, y_cls)
            # loss = ner_loss + cls_loss
            loss = self.config['ner_loss_weight'] * ner_loss + (1 - self.config['ner_loss_weight']) * cls_loss
        elif task_type == TaskType.Joint_mt:
            # cls_loss and mt_loss compute
            ner_logits, mt_logits = self.mnetwork(*inputs)
            y_ner = y_ner.view(-1)
            y_mt = y_mt.view(-1)
            ner_loss = F.cross_entropy(ner_logits, y_ner, ignore_index=-1)
            mt_loss = F.cross_entropy(mt_logits, y_mt, ignore_index=-1)
            loss = self.config['ner_loss_weight'] * ner_loss + (1 - self.config['ner_loss_weight']) * mt_loss
        elif task_type == TaskType.Joint_mt_three:
            ner_logits, mt_logits, cls_logits = self.mnetwork(*inputs)
            y_ner = y_ner.view(-1)
            y_mt = y_mt.view(-1)
            ner_loss = F.cross_entropy(ner_logits, y_ner, ignore_index=-1)
            mt_loss = F.cross_entropy(mt_logits, y_mt, ignore_index=-1)
            cls_loss = F.cross_entropy(cls_logits, y_cls)
            loss = self.config['ner_loss_weight'] * ner_loss + self.config['mt_loss_weight'] * mt_loss + \
                   (1 - self.config['ner_loss_weight'] - self.config['mt_loss_weight']) * cls_loss
        elif task_type == TaskType.Joint_cls_three:
            # cls_loss and mt_loss compute
            ner_logits, bcls_logits, mcls_logits = self.mnetwork(*inputs)
            y_ner = y_ner.view(-1)
            ner_loss = F.cross_entropy(ner_logits, y_ner, ignore_index=-1)
            bcls_loss = F.cross_entropy(bcls_logits, y_bcls)
            mcls_loss = F.cross_entropy(mcls_logits, y_mcls)
            loss = self.config['ner_loss_weight'] * ner_loss + self.config['bcls_loss_weight'] * bcls_loss + \
                   self.config['mcls_loss_weight'] * mcls_loss
        elif task_type == TaskType.Joint_all:
            ner_logits, mt_logits, bcls_logits, mcls_logits = self.mnetwork(*inputs)
            y_ner = y_ner.view(-1)
            ner_loss = F.cross_entropy(ner_logits, y_ner, ignore_index=-1)
            mt_ner = y_mt.view(-1)
            mt_loss = F.cross_entropy(mt_logits, mt_ner, ignore_index=-1)
            bcls_loss = F.cross_entropy(bcls_logits, y_bcls)
            mcls_loss = F.cross_entropy(mcls_logits, y_mcls)
            loss = self.config['ner_loss_weight'] * ner_loss + self.config['mt_loss_weight'] * mt_loss + \
                   self.config['bcls_loss_weight'] * bcls_loss + self.config['mcls_loss_weight'] * mcls_loss
        elif task_type == TaskType.Joint_bCLS_mCLS:
            bcls_logits, mcls_logits = self.mnetwork(*inputs)
            bcls_loss = F.cross_entropy(bcls_logits, y_bcls)
            mcls_loss = F.cross_entropy(mcls_logits, y_mcls)
            loss = self.config['bcls_loss_weight'] * bcls_loss + (1 - self.config['bcls_loss_weight']) * mcls_loss
        elif task_type == TaskType.ReadingComprehension:
            # loss 计算
            start_logits, end_logits, span_logits = self.mnetwork(*inputs)
            start_loss, end_loss, match_loss = mrc_loss_compute(
                start_logits=start_logits,
                end_logits=end_logits,
                span_logits=span_logits,
                start_labels=start_labels,
                end_labels=end_labels,
                match_label=match_labels,
                start_label_mask=start_label_mask,
                end_label_mask=end_label_mask,
                span_loss_candidates='pred_and_gold',
                loss_type='bce'
            )
            # loss = (start_loss + end_loss + match_loss) / 3
            loss = start_loss + end_loss + match_loss

        elif task_type == TaskType.SequenceLabeling:
            # y = labels, 如果是torch.view(-1)，则原张量会变成一维的结构。
            if self.config['use_crf']:
                # 使用crf
                logits, crf_layer = self.mnetwork(*inputs)
                loss = -1 * (crf_layer(emissions=logits, tags=labels, mask=batch_data[batch_meta['mask']]))
            else:
                # 不使用CRF但使用cache
                if self.config['use_cache']:
                    y_ner, y_cache = self.convert_y(y)
                    # y_ner = y.view(-1)
                    ner_logits, cache_logits = self.mnetwork(*inputs)
                    loss_ner = F.cross_entropy(ner_logits, y_ner, ignore_index=-1)
                    loss_cache = F.mse_loss(cache_logits, y_cache)
                    loss = self.config['ner_loss_weight'] * loss_ner + (1 - self.config['ner_loss_weight']) * loss_cache
                else:
                    # 不使用crf也不使用cache
                    y = y.view(-1)
                    logits = self.mnetwork(*inputs)
                    # only real label ids contribute to the loss update
                    loss = F.cross_entropy(logits, y, ignore_index=-1)
        else:
            # for classification
            # len(inputs) = 6, type(inputs) = list
            # 在参数名之前使用一个星号，就是让函数接受任意多的位置参数。
            logits = self.mnetwork(*inputs)
            if task_type == TaskType.Ranking:
                logits = logits.view(-1, batch_meta['pairwise_size'])
            if self.config.get('weighted_on', False):
                if task_type == TaskType.Regression:
                    loss = torch.mean(F.mse_loss(logits.squeeze(), y, reduce=False) * weight)
                else:
                    loss = torch.mean(F.cross_entropy(logits, y, reduce=False) * weight)
                    if soft_labels is not None:
                        # compute KL
                        label_size = soft_labels.size(1)
                        kd_loss = F.kl_div(F.log_softmax(logits.view(-1, label_size).float(), 1), soft_labels,
                                           reduction='batchmean')
                        loss = loss + kd_loss
            else:
                if task_type == TaskType.Regression:
                    loss = F.mse_loss(logits.squeeze(), y)
                else:
                    loss = F.cross_entropy(logits, y)
                    if soft_labels is not None:
                        # compute KL
                        label_size = soft_labels.size(1)
                        kd_loss = F.kl_div(F.log_softmax(logits.view(-1, label_size).float(), 1), soft_labels,
                                           reduction='batchmean')
                        loss = loss + kd_loss

        self.train_loss.update(loss.item(), batch_data[batch_meta['token_id']].size(0))
        # scale loss
        loss = loss / self.config.get('grad_accumulation_step', 1)
        if self.config['fp16']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.get('grad_accumulation_step', 1) == 0:
            if self.config['global_grad_clipping'] > 0:
                if self.config['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                                   self.config['global_grad_clipping'])
                else:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                   self.config['global_grad_clipping'])

            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.update_ema()

    def predict(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta['task_id']
        task_type = batch_meta['task_type']
        inputs = batch_data[:batch_meta['input_len']]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        # return start_scores, end_scores, span_logits for reading comprehension
        score = self.mnetwork(*inputs)
        if task_type == TaskType.Ranking:
            score = score.contiguous().view(-1, batch_meta['pairwise_size'])
            assert task_type == TaskType.Ranking
            score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta['true_label']
        elif task_type == TaskType.SequenceLabeling:
            mask = batch_data[batch_meta['mask']]
            if self.config['use_crf']:
                # score size [batch_size, seq_len, num_labels]
                assert len(score) == 2
                crf = score[-1]
                score = score[0].contiguous()
                tags = crf.decode(score, mask)  # [n_best, batch_size, seq_len]
                tags = tags.squeeze(0).cpu().numpy().tolist()
                valied_length = mask.sum(1).tolist()
                final_predict = []
                for idx, label in enumerate(tags):
                    final_predict.append(label[: valied_length[idx]])
                score = score.data.cpu()
                score = score.numpy()
                score = score.reshape(-1).tolist()
                return score, final_predict, batch_meta['label']
            else:
                if self.config['use_cache']:
                    score = score[0].contiguous()
                else:
                    score = score.contiguous()
                score = score.data.cpu()
                score = score.numpy()
                predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
                valied_length = mask.sum(1).tolist()
                final_predict = []
                for idx, p in enumerate(predict):
                    final_predict.append(p[: valied_length[idx]])
                score = score.reshape(-1).tolist()
                return score, final_predict, batch_meta['label']
        elif task_type == TaskType.ReadingComprehension:
            assert len(score) == 3
            start_preds, end_preds = score[0] > 0, score[1] > 0
            match_logits = score[-1]
            start_label_mask = batch_data[batch_meta['start_label_mask']].cuda(non_blocking=True)
            end_label_mask = batch_data[batch_meta['end_label_mask']].cuda(non_blocking=True)
            match_labels = batch_data[batch_meta['match_labels']].cuda(non_blocking=True)

            # compute span f1 according to query-based model output
            start_label_mask = start_label_mask.bool()
            end_label_mask = end_label_mask.bool()
            match_labels = match_labels.bool()
            bsz, seq_len = start_label_mask.size()
            match_preds = match_logits > 0
            start_preds = start_preds.bool()
            end_preds = end_preds.bool()
            match_preds = (match_preds
                           & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                           & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
            match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                                & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
            match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
            match_preds = match_label_mask & match_preds

            return batch_meta['answer'], match_preds, match_labels,
        elif task_type == TaskType.Joint:
            # todo 精简代码
            mask = batch_data[batch_meta['mask']]
            assert len(score) == 2
            ner_score, cls_score = score[0], score[-1]
            predict = []
            gold = []
            final_score = []
            # ner
            ner_score = ner_score.contiguous()
            ner_score = ner_score.data.cpu()
            ner_score = ner_score.numpy()
            ner_predict = np.argmax(ner_score, axis=1).reshape(mask.size()).tolist()
            valied_length = mask.sum(1).tolist()
            ner_final_predict = []
            for idx, p in enumerate(ner_predict):
                ner_final_predict.append(p[: valied_length[idx]])
            ner_score = ner_score.reshape(-1).tolist()
            # cls
            cls_score = F.softmax(cls_score, dim=1)
            cls_score = cls_score.data.cpu()
            cls_score = cls_score.numpy()
            cls_predict = np.argmax(cls_score, axis=1).tolist()
            cls_score = cls_score.reshape(-1).tolist()
            final_score.append(ner_score)
            final_score.append(cls_score)
            predict.append(ner_final_predict)
            predict.append(cls_predict)
            gold.append(batch_meta['ner_label'])
            gold.append(batch_meta['cls_label'])
            return final_score, predict, gold
        elif task_type == TaskType.Joint_mt:
            mask = batch_data[batch_meta['mask']]
            assert len(score) == 2
            ner_score, mt_score = score[0], score[-1]
            ner_score, mt_score = ner_score.contiguous(), mt_score.contiguous()
            ner_score, mt_score = ner_score.data.cpu(), mt_score.data.cpu()
            ner_score, mt_score = ner_score.numpy(), mt_score.numpy()
            ner_predict = np.argmax(ner_score, axis=1).reshape(mask.size()).tolist()
            mt_predict = np.argmax(mt_score, axis=1).reshape(mask.size()).tolist()
            valied_length = mask.sum(1).tolist()
            ner_final_predict, mt_final_predict = [], []
            for idx, p in enumerate(ner_predict):
                ner_final_predict.append(p[: valied_length[idx]])
            for idx, p in enumerate(mt_predict):
                mt_final_predict.append(p[: valied_length[idx]])
            ner_score, mt_score = ner_score.reshape(-1).tolist(), mt_score.reshape(-1).tolist()
            final_score, final_predict, final_gold = [], [], []
            final_score.append(ner_score)
            final_score.append(mt_score)
            final_predict.append(ner_final_predict)
            final_predict.append(mt_final_predict)
            final_gold.append(batch_meta['ner_label'])
            final_gold.append(batch_meta['mt_label'])
            return final_score, final_predict, final_gold
        elif task_type == TaskType.Joint_mt_three:
            mask = batch_data[batch_meta['mask']]
            assert len(score) == 3
            ner_score, mt_score, cls_score = score[0], score[1], score[2]
            cls_score = F.softmax(cls_score, dim=1)
            cls_score = cls_score.data.cpu()
            cls_score = cls_score.numpy()
            cls_predict = np.argmax(cls_score, axis=1).tolist()
            cls_score = cls_score.reshape(-1).tolist()
            ner_score, mt_score = ner_score.contiguous(), mt_score.contiguous()
            ner_score, mt_score = ner_score.data.cpu(), mt_score.data.cpu()
            ner_score, mt_score = ner_score.numpy(), mt_score.numpy()
            ner_predict = np.argmax(ner_score, axis=1).reshape(mask.size()).tolist()
            mt_predict = np.argmax(mt_score, axis=1).reshape(mask.size()).tolist()
            valied_length = mask.sum(1).tolist()
            ner_final_predict, mt_final_predict = [], []
            for idx, p in enumerate(ner_predict):
                ner_final_predict.append(p[: valied_length[idx]])
            for idx, p in enumerate(mt_predict):
                mt_final_predict.append(p[: valied_length[idx]])
            ner_score, mt_score = ner_score.reshape(-1).tolist(), mt_score.reshape(-1).tolist()
            final_score, final_predict, final_gold = [], [], []
            final_score.append(ner_score)
            final_score.append(mt_score)
            final_score.append(cls_score)
            final_predict.append(ner_final_predict)
            final_predict.append(mt_final_predict)
            final_predict.append(cls_predict)
            final_gold.append(batch_meta['ner_label'])
            final_gold.append(batch_meta['mt_label'])
            final_gold.append(batch_meta['cls_label'])
            return final_score, final_predict, final_gold
        elif task_type == TaskType.Joint_cls_three:
            mask = batch_data[batch_meta['mask']]
            assert len(score) == 3
            ner_score, bcls_score, mcls_score = score[0], score[1], score[2]
            predict = []
            gold = []
            final_score = []
            # ner
            ner_score = ner_score.contiguous()
            ner_score = ner_score.data.cpu()
            ner_score = ner_score.numpy()
            ner_predict = np.argmax(ner_score, axis=1).reshape(mask.size()).tolist()
            valied_length = mask.sum(1).tolist()
            ner_final_predict = []
            for idx, p in enumerate(ner_predict):
                ner_final_predict.append(p[: valied_length[idx]])
            ner_score = ner_score.reshape(-1).tolist()
            # bcls
            bcls_score = F.softmax(bcls_score, dim=1)
            bcls_score = bcls_score.data.cpu()
            bcls_score = bcls_score.numpy()
            bcls_predict = np.argmax(bcls_score, axis=1).tolist()
            bcls_score = bcls_score.reshape(-1).tolist()
            # mcls
            mcls_score = F.softmax(mcls_score, dim=1)
            mcls_score = mcls_score.data.cpu()
            mcls_score = mcls_score.numpy()
            mcls_predict = np.argmax(mcls_score, axis=1).tolist()
            mcls_score = mcls_score.reshape(-1).tolist()
            final_score.append(ner_score)
            final_score.append(bcls_score)
            final_score.append(mcls_score)
            predict.append(ner_final_predict)
            predict.append(bcls_predict)
            predict.append(mcls_predict)
            gold.append(batch_meta['ner_label'])
            gold.append(batch_meta['bcls_label'])
            gold.append(batch_meta['mcls_label'])
            return final_score, predict, gold
        elif task_type == TaskType.Joint_all:
            mask = batch_data[batch_meta['mask']]
            assert len(score) == 4
            ner_score, mt_score, bcls_score, mcls_score = score[0], score[1], score[2], score[3]
            predict = []
            gold = []
            final_score = []
            # ner
            ner_score = ner_score.contiguous()
            ner_score = ner_score.data.cpu()
            ner_score = ner_score.numpy()
            ner_predict = np.argmax(ner_score, axis=1).reshape(mask.size()).tolist()
            valied_length = mask.sum(1).tolist()
            ner_final_predict = []
            for idx, p in enumerate(ner_predict):
                ner_final_predict.append(p[: valied_length[idx]])
            ner_score = ner_score.reshape(-1).tolist()
            # mt
            mt_score = mt_score.contiguous()
            mt_score = mt_score.data.cpu()
            mt_score = mt_score.numpy()
            mt_predict = np.argmax(mt_score, axis=1).reshape(mask.size()).tolist()
            valied_length = mask.sum(1).tolist()
            mt_final_predict = []
            for idx, p in enumerate(mt_predict):
                mt_final_predict.append(p[: valied_length[idx]])
            mt_score = mt_score.reshape(-1).tolist()
            # bcls
            bcls_score = F.softmax(bcls_score, dim=1)
            bcls_score = bcls_score.data.cpu()
            bcls_score = bcls_score.numpy()
            bcls_predict = np.argmax(bcls_score, axis=1).tolist()
            bcls_score = bcls_score.reshape(-1).tolist()
            # mcls
            mcls_score = F.softmax(mcls_score, dim=1)
            mcls_score = mcls_score.data.cpu()
            mcls_score = mcls_score.numpy()
            mcls_predict = np.argmax(mcls_score, axis=1).tolist()
            mcls_score = mcls_score.reshape(-1).tolist()
            final_score.append(ner_score)
            final_score.append(mt_score)
            final_score.append(bcls_score)
            final_score.append(mcls_score)
            predict.append(ner_final_predict)
            predict.append(mt_final_predict)
            predict.append(bcls_predict)
            predict.append(mcls_predict)
            gold.append(batch_meta['ner_label'])
            gold.append(batch_meta['mt_label'])
            gold.append(batch_meta['bcls_label'])
            gold.append(batch_meta['mcls_label'])
            return final_score, predict, gold
        elif task_type == TaskType.Joint_bCLS_mCLS:
            assert len(score) == 2
            bcls_score, mcls_score = score[0], score[1]
            predict = []
            gold = []
            final_score = []
            # bcls
            bcls_score = F.softmax(bcls_score, dim=1)
            bcls_score = bcls_score.data.cpu()
            bcls_score = bcls_score.numpy()
            bcls_predict = np.argmax(bcls_score, axis=1).tolist()
            bcls_score = bcls_score.reshape(-1).tolist()
            # mcls
            mcls_score = F.softmax(mcls_score, dim=1)
            mcls_score = mcls_score.data.cpu()
            mcls_score = mcls_score.numpy()
            mcls_predict = np.argmax(mcls_score, axis=1).tolist()
            mcls_score = mcls_score.reshape(-1).tolist()
            final_score.append(bcls_score)
            final_score.append(mcls_score)
            predict.append(bcls_predict)
            predict.append(mcls_predict)
            gold.append(batch_meta['bcls_label'])
            gold.append(batch_meta['mcls_label'])
            return final_score, predict, gold
        else:
            if task_type == TaskType.Classification:
                score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).tolist()
            score = score.reshape(-1).tolist()
        return score, predict, batch_meta['label']

    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def save(self, filename):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        ema_state = dict(
            [(k, v.cpu()) for k, v in self.ema.model.state_dict().items()]) if self.ema is not None else dict()
        params = {
            'state': network_state,
            'optimizer': self.optimizer.state_dict(),
            'ema': ema_state,
            'config': self.config,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def load(self, checkpoint):

        model_state_dict = torch.load(checkpoint)
        if model_state_dict['config']['init_checkpoint'].rsplit('/', 1)[1] != \
                self.config['init_checkpoint'].rsplit('/', 1)[1]:
            logger.error(
                '*** SANBert network is pretrained on a different Bert Model. Please use that to fine-tune for other tasks. ***')
            sys.exit()

        self.network.load_state_dict(model_state_dict['state'], strict=False)
        self.optimizer.load_state_dict(model_state_dict['optimizer'])
        self.config = model_state_dict['config']
        if self.ema:
            self.ema.model.load_state_dict(model_state_dict['ema'])

    def cuda(self):
        self.network.cuda()
        if self.config['ema_opt']:
            self.ema.cuda()

    def convert_y(self, y):
        """
        用ner标签来指导cache, 比如希望cache存储含有entity的句子-> bCSL
        希望cache存储含有multi-token entity的句子-> mtCLS
        :param y: batch_size * seq_len
        :return:
        """
        batch_size, seq_len = y.size()
        y_cache = torch.FloatTensor(batch_size)
        # y_cache = torch.FloatTensor(batch_size, seq_len)
        if self.config['cuda']:
            y_cache = y_cache.cuda(non_blocking=True)
        y_cache.requires_grad = False
        for idx in range(batch_size):
            if 1 in y[idx] or 2 in y[idx] or 3 in y[idx] or 4 in y[idx] or 5 in y[idx] or 6 in y[idx]:
                y_cache[idx] = 1
            else:
                y_cache[idx] = 0
        y_ner = y.view(-1)
        return y_ner, y_cache
