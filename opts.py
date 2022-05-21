"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-16
"""
import torch
import argparse
from data_utils.task_def import EncoderModelType


def model_config(parser):
    """
    :param parser:
    :return:
    """
    parser.add_argument('--update_bert_opt', default=0, type=int)
    # 顾名思义，store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False。
    # 反之亦然，store_false也就是默认为True，一旦命令中有此参数，其值则变为False。
    # todo 给argument增加help参数
    parser.add_argument('--multi_gpu_on', action='store_true')
    parser.add_argument('--mem_cum_type', type=str, default='simple', help='bilinear/simple/default')
    parser.add_argument('--answer_num_turn', type=int, default=5)
    parser.add_argument('--answer_mem_drop_p', type=float, default=0.1)
    parser.add_argument('--answer_att_hidden_size', type=int, default=128)
    parser.add_argument('--answer_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_rnn_type', type=str, default='gru',
                        help='rnn/gru/lstm')
    parser.add_argument('--answer_sum_att_type', type=str, default='bilinear',
                        help='bilinear/simple/defualt')
    parser.add_argument('--answer_merge_opt', type=int, default=1)
    parser.add_argument('--answer_mem_type', type=int, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_weight_norm_on', action='store_true')
    parser.add_argument('--dump_state_on', action='store_true')
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='3')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--mix_opt', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--encoder_type', type=int, default=EncoderModelType.BERT)
    # for NER
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument('--crf_learning_rate', type=float, default=5e-5)
    # for MRC
    parser.add_argument('--mrc_dropout', type=float, default=0.1)
    # for cache
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--cache_size', type=int, default=32)
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--rnn_hidden_size', type=int, default=512)
    parser.add_argument('--rnn_num_layers', type=int, default=2)
    parser.add_argument('--rnn_dropout', type=int, default=0.1)
    parser.add_argument('--query_and_read_strategy', type=str, default='context_gate',
                        choices=['normal_add', 'normal_cat', 'context_gate', 'DNN'])
    parser.add_argument('--cache_pooling', type=str, default='CLS', choices=['MAX', 'MEAN', 'MEAN_sqrt', 'CLS'])
    parser.add_argument('--cache_update_strategy', type=str, default='score', choices=['FIFO', 'LFU', 'LRU', 'score'])
    parser.add_argument('--cache_query_strategy', type=str, default='dot', choices=['dot', 'Man', 'Euc', 'cos'])
    # for classification
    parser.add_argument('--cls_pooling', type=str, default='CLS', choices=['CLS', 'MAX', 'MEAN', 'MEAN_sqrt'])
    parser.add_argument('--use_luo', action='store_true')

    return parser


def data_config(parser):
    """
    :param parser:
    :return:
    """
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument('--tensorboard', default='store_true')
    parser.add_argument('--tensorboard_logdir', default='tensorboard_logdir')
    parser.add_argument('--init_checkpoint', default='pretrained_models/biobert_base_cased/pytorch_model.pt', type=str)
    parser.add_argument('--data_dir', default='canonical_data/bert_cased')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--task_def', type=str, default='multi_task_def.yml')
    parser.add_argument('--train_datasets', default='NCBI-disease-IOB')
    parser.add_argument('--test_datasets', default='NCBI-disease-IOB')
    return parser


def train_config(parser):
    """
    :param parser:
    :return:
    """
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--save_per_updates', type=int, default=10000)
    parser.add_argument('--save_per_updates_on', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--optimizer', default='adam',
                        help='supported optimizer: adamax, sgd, adadelta, adam')
    parser.add_argument('--grad_clipping', type=float, default=1.0)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--ner_loss_weight', type=float, default=0.7)
    parser.add_argument('--bcls_loss_weight', type=float, default=0.3)
    parser.add_argument('--mcls_loss_weight', type=float, default=0.3)
    parser.add_argument('--mt_loss_weight', type=float, default=0.3)

    parser.add_argument('--vb_dropout', action='store_false')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--dropout_w', type=float, default=0.000,
                        help='Randomly drop a fraction drooput_w of training instances.')
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)

    # loading
    parser.add_argument("--model_ckpt", default='checkpoints/model_0.pt', type=str)
    parser.add_argument("--resume", action='store_true')

    # EMA
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--ema_gamma', type=float, default=0.995)

    # scheduler
    parser.add_argument('--have_lr_scheduler', dest='have_lr_scheduler', action='store_false')
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--freeze_layers', type=int, default=-1)
    parser.add_argument('--embedding_opt', type=int, default=0)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--bert_l2norm', type=float, default=0.0)
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--grad_accumulation_step', type=int, default=1)

    # fp 16
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # save
    parser.add_argument('--not_save', action='store_true', help="Don't save the model")
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    options = get_args()
