"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-18
"""
import torch
import json
import copy
import os
import random
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from mt_dnn.mt_dnn_utils import set_environment
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDnnModel
from data_utils.logger_wrapper import create_logger
from multi_exp_def import MultiTaskDefs
from opts import get_args
from mt_dnn.inference import *
from pytorch_pretrained_bert.modeling import BertConfig


def dump(path, data):
    """
    :param path:
    :param data:
    :return:
    """
    with open(path, 'w') as fa:
        json.dump(data, fa)


def dump2(path, uids, golds, predictions):
    """
    :param path:
    :param uids:
    :param golds:
    :param predictions:
    :return:
    """
    with open(path, 'w') as fa:
        for uid, score, pred in zip(uids, golds, predictions):
            s = json.dumps({'uid': uid, 'golds': score, 'prediction': pred})
            fa.write(s + '\n')


def generate_decoder_opt(enable_san, max_opt):
    """
    :param enable_san:
    :param max_opt:
    :return:
    """
    opt_v = 0
    if enable_san and max_opt < 3:
        opt_v = max_opt
    return opt_v


def train():
    args = get_args()
    #args.use_cache = True
    args.train_datasets = args.train_datasets.split(',')
    args.test_datasets = args.test_datasets.split(',')
    args.output_dir = os.path.join(os.getcwd(), args.output_dir)

    logger = create_logger(__name__, to_disk=True, log_file=args.log_file)
    if os.path.isdir(args.output_dir):
        logger.warning('path %s exists', args.output_dir)
    else:
        os.mkdir(args.output_dir)

    logger.info('args: %s', json.dumps(vars(args), indent=2))
    # print user config
    set_environment(args.seed, args.cuda)
    task_defs = MultiTaskDefs(args.task_def)
    encoder_type = task_defs.encoder_type
    args.encoder_type = encoder_type

    logger.info('Launching the MT-DNN training')

    # update data dir
    train_data_list = []
    nclass_list = []
    decoder_opts = []
    task_types = []
    dropout_list = []
    tasks = {}
    tasks_class = {}

    for dataset in args.train_datasets:
        task = dataset.split('_')[0]
        if task in tasks:
            logger.warning('Skipping: %s in %s', task, tasks)
            continue
        task_id = len(tasks)
        data_type = task_defs.data_format_map[task]
        task_type = task_defs.task_type_map[task]
        if 'Joint-two' in task or 'Joint-bCLS-mtCLS' in task or 'Joint-mCLS-mtCLS' in task:
            assert task in task_defs.ner_n_class_map
            assert task in task_defs.cls_n_class_map
            # todo n_clss这个参数有点冗余
            nclass = [task_defs.ner_n_class_map[task], task_defs.cls_n_class_map[task]]
        elif 'Joint-bCLS-mCLS' in task:
            assert task in task_defs.bcls_n_class_map
            assert task in task_defs.mcls_n_class_map
            nclass = [task_defs.bcls_n_class_map[task], task_defs.mcls_n_class_map[task]]
        elif 'Joint-mt' in task:
            assert task in task_defs.ner_n_class_map
            assert task in task_defs.mt_n_class_map
            nclass = [task_defs.ner_n_class_map[task], task_defs.mt_n_class_map[task]]
        elif 'Joint-three-mt' in task:
            assert task in task_defs.ner_n_class_map
            assert task in task_defs.mt_n_class_map
            assert task in task_defs.cls_n_class_map
            nclass = [task_defs.ner_n_class_map[task], task_defs.mt_n_class_map[task], task_defs.cls_n_class_map[task]]
        elif 'Joint-three-cls' in task:
            assert task in task_defs.ner_n_class_map
            assert task in task_defs.bcls_n_class_map
            assert task in task_defs.mcls_n_class_map
            nclass = [task_defs.ner_n_class_map[task], task_defs.bcls_n_class_map[task], task_defs.mcls_n_class_map[task]]
        elif 'Joint-all' in task:
            assert task in task_defs.ner_n_class_map
            assert task in task_defs.bcls_n_class_map
            assert task in task_defs.mcls_n_class_map
            assert task in task_defs.mt_n_class_map
            nclass = [task_defs.ner_n_class_map[task], task_defs.mt_n_class_map[task],
                      task_defs.bcls_n_class_map[task], task_defs.mcls_n_class_map[task]]
        elif 'MRC-rule' in task or 'MRC-wiki' in task:
            assert task in task_defs.ner_n_class_map
            assert task in task_defs.mrc_n_class_map
            nclass = [task_defs.ner_n_class_map[task], task_defs.mrc_n_class_map[task]]
        else:
            assert task in task_defs.n_class_map
            nclass = task_defs.n_class_map[task]
            if args.mtl_opt > 0:
                # default = 0
                task_id = tasks_class[nclass] if nclass in tasks_class else len(tasks_class)

        dopt = generate_decoder_opt(task_defs.enable_san_map[task], args.answer_opt)
        if task_id < len(decoder_opts):
            decoder_opts[task_id] = min(decoder_opts[task_id], dopt)
        else:
            decoder_opts.append(dopt)
        task_types.append(task_type)

        if task not in tasks:
            tasks[task] = len(tasks)  # task_id
            if args.mtl_opt < 1:
                nclass_list.append(nclass)

        if 'Joint' not in dataset and 'MRC-rule' not in dataset and 'MRC-wiki' not in dataset:
            if nclass not in tasks_class:
                tasks_class[nclass] = len(tasks_class)
                if args.mtl_opt > 0:
                    nclass_list.append(nclass)


        dropout_p = task_defs.dropout_p_map.get(task, args.dropout_p)  # get(key, default=)
        dropout_list.append(dropout_p)

        # use train and dev
        train_path = os.path.join(args.data_dir, f'{task}_train_dev.json')
        logger.info('Loading %s as task %s', task, task_id)
        train_data = BatchGen(
            data=BatchGen.load(train_path, True, task_type=task_type, maxlen=args.max_seq_len),
            batch_size=args.batch_size,
            dropout_w=args.dropout_w,
            gpu=args.cuda,
            task_id=task_id,
            maxlen=args.max_seq_len,
            data_type=data_type,
            task_type=task_type,
            encoder_type=encoder_type
        )
        train_data_list.append(train_data)

    dev_data_list = []
    test_data_list = []

    for dataset in args.test_datasets:
        task = dataset.split('_')[0]
        # print(task)
        if 'Joint' in task or 'MRC' in task:
            task_id = tasks[task]
        else:
            task_id = tasks_class[task_defs.n_class_map[task]] if args.mtl_opt > 0 else tasks[task]
        task_type = task_defs.task_type_map[task]
        data_type = task_defs.data_format_map[task]

        dev_path = os.path.join(args.data_dir, f'{dataset}_dev.json')
        dev_data = BatchGen(
            data=BatchGen.load(dev_path, False, task_type=task_type, maxlen=args.max_seq_len),
            batch_size=args.batch_size_eval,
            gpu=args.cuda,
            is_train=False,
            task_id=task_id,
            maxlen=args.max_seq_len,
            data_type=data_type,
            task_type=task_type,
            encoder_type=encoder_type
        )
        dev_data_list.append(dev_data)

        test_path = os.path.join(args.data_dir, f'{dataset}_test.json')
        test_data = BatchGen(
            data=BatchGen.load(test_path, False, task_type=task_type, maxlen=args.max_seq_len),
            batch_size=args.batch_size_eval,
            gpu=args.cuda,
            is_train=False,
            task_id=task_id,
            maxlen=args.max_seq_len,
            data_type=data_type,
            task_type=task_type,
            encoder_type=encoder_type
        )
        test_data_list.append(test_data)

    opt = copy.deepcopy(vars(args))
    opt['answer_opt'] = decoder_opts
    opt['task_types'] = task_types
    opt['tasks_dropout_p'] = dropout_list

    label_size = ';'.join([str(l) for l in nclass_list])
    opt['label_size'] = label_size

    logger.info('#' * 20)
    logger.info('opt: %s', json.dumps(opt, indent=2))
    logger.info('#' * 20)

    bert_model_path = args.init_checkpoint
    # state_dict是一个OrderDict，存储了网络结构的名字和对应的参数
    state_dict = None

    if os.path.exists(bert_model_path):
        state_dict = torch.load(bert_model_path)
        # 加载预训练模型
        if 'scibert' in bert_model_path:
            with open('pretrained_models/scibert_scivocab_uncased/config.json', 'r', encoding='utf-8') as fa:
                config = json.load(fa)
        elif 'pubmedbert' in bert_model_path:
            with open('pretrained_models/pubmedbert_base_uncased/config.json', 'r', encoding='utf-8') as fa:
                config = json.load(fa)
        elif 'chinesebert_large' in bert_model_path:
            with open('pretrained_models/chinesebert_large_uncased/config.json', 'r', encoding='utf-8') as fa:
                config = json.load(fa)
        else:
            config = state_dict['config']
        # 加载预训练模型的配置
        config['attention_probs_dropout_prob'] = args.bert_dropout_p
        config['hidden_dropout_prob'] = args.bert_dropout_p
        opt.update(config)
    else:
        logger.error('#' * 20)
        logger.error('Could not find the init model!\n'
                     'The parameters will be initialized randomly!')
        logger.error('#' * 20)
        # 30522是bert_uncased的词表大小
        config = BertConfig(vocab_size_or_config_json_file=30522).to_dict()
        opt.update(config)

    all_iters = [iter(item) for item in train_data_list]
    all_lens = [len(item) for item in train_data_list]
    # all_lens = task1_numbers_of_batches + task2_numbers_of_batches + ...
    # numbers_of_batches = len(sentences) / batch_size
    # div number of grad accumulation.
    num_all_batches = args.epochs * sum(all_lens) // args.grad_accumulation_step
    logger.info('############### Gradient Accumulation Info ###############')
    logger.info('number of step: %s', args.epochs * sum(all_lens))
    logger.info('number of grad_accumulation step: %s', args.grad_accumulation_step)
    logger.info('adjusted number of step: %s', num_all_batches)
    logger.info('############### Gradient Accumulation Info ###############')

    if len(train_data_list) > 1 and args.ratio > 0:
        num_all_batches = int(args.epochs * (len(train_data_list[0]) * (1 + args.ratio)))

    model = MTDnnModel(opt, state_dict=state_dict, num_train_step=num_all_batches)

    if args.resume and args.model_ckpt:
        logger.info('loading model from %s', args.model_ckpt)
        model.load(args.model_ckpt)

    # model meta str
    headline = "########## Model Arch of MT-DNN ##########"
    # print network
    logger.debug('\n{}\n{}\n'.format(headline, model.network))

    # dump config
    config_file = os.path.join(args.output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as fa:
        fa.write('{}\n'.format(json.dumps(opt)))
        fa.write('\n{}\n{}\n'.format(headline, model.network))

    logger.info('Total number of params: %s', model.total_param)

    # tensorboard
    if args.tensorboard:
        args.tensorboard_logdir = os.path.join(args.output_dir, args.tensorboard_logdir)
        tensorboard = SummaryWriter(log_dir=args.tensorboard_logdir)

    for epoch in range(0, args.epochs):
        logger.warning('At epoch %s', epoch)
        for train_data in train_data_list:
            train_data.reset()
        start = datetime.now()
        all_indices = []
        if len(train_data_list) > 1 and args.ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(min(len(train_data_list[0]) * args.ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if args.mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()
        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if args.mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])

        if args.mix_opt < 1:
            random.shuffle(all_indices)

        # all_indices = 一个epoch的training step, shuffled
        for i in range(len(all_indices)):
            task_id = all_indices[i]
            batch_meta, batch_data = next(all_iters[task_id])
            # print(batch_meta) dict
            # print(batch_data) list, [tensor(token_id), tensor(segment_id), tensor(mask_id), tensor(label)]
            model.update(batch_meta, batch_data)
            if model.local_updates % (args.log_per_updates * args.grad_accumulation_step) == 0 \
                    or model.local_updates == 1:
                # 每500个step打印一次
                remaining_time = str(
                    (datetime.now() - start) / (i + 1) * (len(all_indices) - i - 1)
                ).split('.')[0]
                logger.info('Task [%2d] updates[%6d] train loss[%.5f] estimated_remaining_time[%s]',
                            task_id, model.updates, model.train_loss.avg, remaining_time)

                if args.tensorboard:
                    tensorboard.add_scalar('train/loss', model.train_loss.avg,
                                           global_step=model.updates)

            if args.save_per_updates_on \
                    and (model.local_updates % (
                    args.save_per_updates * args.grad_accumulation_step) == 0):
                model_file = os.path.join(args.output_dir, f'model_{epoch}_{model.updates}.pt')
                logger.info('Saving mt-dnn model to %s', model_file)
                model.save(model_file)
        if args.use_cache:
            # 缓存清空
            model.mnetwork.dynamic_cache.key = []
            model.mnetwork.dynamic_cache.value = []
            model.mnetwork.dynamic_cache.score = []
            model.mnetwork.dynamic_cache.word_cache = []
            model.mnetwork.dynamic_cache.cnt = []
            model.mnetwork.dynamic_cache.timestep = []
            model.mnetwork.dynamic_cache.current_timestep = 0

            model.mnetwork.dynamic_cache.oldest_index = 0
        # logger.info('Task %s - epoch %s - cache_score_list %s', dataset, epoch, str(model.mnetwork.dynamic_cache.score))
        for idx, dataset in enumerate(args.test_datasets):
            task = dataset.split('_')[0]
            # todo 简化代码, 移出train函数
            if 'Joint-two' in dataset or 'Joint-bCLS-mtCLS' in dataset or 'Joint-mCLS-mtCLS' in task:
                ner_label_mapper = task_defs.ner_label_mapper_map[task]
                cls_label_mapper = task_defs.cls_label_mapper_map[task]
                ner_metric_meta = task_defs.ner_metric_meta_map[task]
                cls_metric_meta = task_defs.cls_metric_meta_map[task]
                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    cls_metrics, cls_predictions, cls_scores, cls_golds, ids = eval_joint_model(
                        model, data, ner_metric_meta, cls_metric_meta, args.cuda, True, ner_label_mapper, cls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in cls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    cls_metrics, cls_predictions, cls_scores, cls_golds, ids = eval_joint_model(
                        model, data, ner_metric_meta, cls_metric_meta, args.cuda, True, ner_label_mapper, cls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in cls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)

            elif 'Joint-mt' in dataset:
                # todo 代码和Joint-two重复太多
                ner_label_mapper = task_defs.ner_label_mapper_map[task]
                mt_label_mapper = task_defs.mt_label_mapper_map[task]
                ner_metric_meta = task_defs.ner_metric_meta_map[task]
                mt_metric_meta = task_defs.mt_metric_meta_map[task]
                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    cls_metrics, cls_predictions, cls_scores, cls_golds, ids = eval_joint_model(
                        model, data, ner_metric_meta, mt_metric_meta, args.cuda, True, ner_label_mapper, mt_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in cls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    cls_metrics, cls_predictions, cls_scores, cls_golds, ids = eval_joint_model(
                        model, data, ner_metric_meta, mt_metric_meta, args.cuda, True, ner_label_mapper, mt_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in cls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
            elif 'Joint-three-mt' in dataset:
                ner_label_mapper = task_defs.ner_label_mapper_map[task]
                mt_label_mapper = task_defs.mt_label_mapper_map[task]
                cls_label_mapper = task_defs.cls_label_mapper_map[task]
                ner_metric_meta = task_defs.ner_metric_meta_map[task]
                mt_metric_meta = task_defs.mt_metric_meta_map[task]
                cls_metric_meta = task_defs.cls_metric_meta_map[task]
                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    mt_metrics, mt_predictions, mt_scores, mt_golds,\
                    cls_metrics, cls_predictions, cls_scores, cls_golds, ids = eval_joint_model_three(
                        model, data, ner_metric_meta, mt_metric_meta, cls_metric_meta, args.cuda, True,
                        ner_label_mapper, mt_label_mapper, cls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in mt_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in cls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    mt_metrics, mt_predictions, mt_scores, mt_golds,\
                    cls_metrics, cls_predictions, cls_scores, cls_golds, ids = eval_joint_model_three(
                        model, data, ner_metric_meta, mt_metric_meta, cls_metric_meta, args.cuda, True,
                        ner_label_mapper, mt_label_mapper, cls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in mt_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in cls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
            elif 'Joint-three-cls' in dataset:
                ner_label_mapper = task_defs.ner_label_mapper_map[task]
                bcls_label_mapper = task_defs.bcls_label_mapper_map[task]
                mcls_label_mapper = task_defs.mcls_label_mapper_map[task]
                ner_metric_meta = task_defs.ner_metric_meta_map[task]
                bcls_metric_meta = task_defs.bcls_metric_meta_map[task]
                mcls_metric_meta = task_defs.mcls_metric_meta_map[task]
                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    bcls_metrics, bcls_predictions, bcls_scores, bcls_golds,\
                    mcls_metrics, mcls_predictions, mcls_scores, mcls_golds, ids = eval_joint_model_three(
                        model, data, ner_metric_meta, bcls_metric_meta, mcls_metric_meta, args.cuda, True,
                        ner_label_mapper, bcls_label_mapper, mcls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in bcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in mcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    bcls_metrics, bcls_predictions, bcls_scores, bcls_golds,\
                    mcls_metrics, mcls_predictions, mcls_scores, mcls_golds, ids = eval_joint_model_three(
                        model, data, ner_metric_meta, bcls_metric_meta, mcls_metric_meta, args.cuda, True,
                        ner_label_mapper, bcls_label_mapper, mcls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in bcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in mcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
            elif 'Joint-all' in dataset:
                ner_label_mapper = task_defs.ner_label_mapper_map[task]
                mt_label_mapper = task_defs.mt_label_mapper_map[task]
                bcls_label_mapper = task_defs.bcls_label_mapper_map[task]
                mcls_label_mapper = task_defs.mcls_label_mapper_map[task]
                ner_metric_meta = task_defs.ner_metric_meta_map[task]
                mt_metric_meta = task_defs.mt_metric_meta_map[task]
                bcls_metric_meta = task_defs.bcls_metric_meta_map[task]
                mcls_metric_meta = task_defs.mcls_metric_meta_map[task]
                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    mt_metrics, mt_predictions, mt_scores, mt_golds, \
                    bcls_metrics, bcls_predictions, bcls_scores, bcls_golds,\
                    mcls_metrics, mcls_predictions, mcls_scores, mcls_golds, ids = eval_joint_model_all(
                        model, data, ner_metric_meta, mt_metric_meta, bcls_metric_meta, mcls_metric_meta, args.cuda, True,
                        ner_label_mapper, mt_label_mapper, bcls_label_mapper, mcls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in mt_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in bcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in mcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    ner_metrics, ner_predictions, ner_scores, ner_golds, \
                    mt_metrics, mt_predictions, mt_scores, mt_golds, \
                    bcls_metrics, bcls_predictions, bcls_scores, bcls_golds,\
                    mcls_metrics, mcls_predictions, mcls_scores, mcls_golds, ids = eval_joint_model_all(
                        model, data, ner_metric_meta, mt_metric_meta, bcls_metric_meta, mcls_metric_meta, args.cuda, True,
                        ner_label_mapper, mt_label_mapper, bcls_label_mapper, mcls_label_mapper
                    )
                for key, val in ner_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in mt_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in bcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in mcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
            elif 'Joint-bCLS-mCLS' in dataset:
                bcls_label_mapper = task_defs.bcls_label_mapper_map[task]
                mcls_label_mapper = task_defs.mcls_label_mapper_map[task]
                bcls_metric_meta = task_defs.bcls_metric_meta_map[task]
                mcls_metric_meta = task_defs.mcls_metric_meta_map[task]
                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    bcls_metrics, bcls_predictions, bcls_scores, bcls_golds,\
                    mcls_metrics, mcls_predictions, mcls_scores, mcls_golds, ids = eval_joint_model(
                        model, data, bcls_metric_meta, mcls_metric_meta, args.cuda, True,
                        bcls_label_mapper, mcls_label_mapper
                    )
                for key, val in bcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                for key, val in mcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    bcls_metrics, bcls_predictions, bcls_scores, bcls_golds,\
                    mcls_metrics, mcls_predictions, mcls_scores, mcls_golds, ids = eval_joint_model(
                        model, data, bcls_metric_meta, mcls_metric_meta, args.cuda, True,
                        bcls_label_mapper, mcls_label_mapper
                    )
                for key, val in bcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
                for key, val in mcls_metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s:  %s', dataset, epoch, key, val)
            elif 'MRC-rule' in dataset or 'MRC-wiki' in dataset:
                ner_label_mapper = task_defs.ner_label_mapper_map[task]
                ner_metric_meta = task_defs.ner_metric_meta_map[task]

                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    metrics, predictions, scores, golds, ids = eval_model(
                        model, data, ner_metric_meta, args.cuda, True, ner_label_mapper)
                for key, val in metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                path = os.path.join(args.output_dir, f'{dataset}_dev_scores_{epoch}.json')
                result = {'metrics': metrics, 'predictions': predictions, 'uids': ids, 'scores': scores}
                # dump(path, result)

                path = os.path.join(args.output_dir, f'{dataset}_dev_scores_{epoch}_2.json')
                # dump2(path, ids, scores, predictions)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    metrics, predictions, scores, golds, ids = eval_model(
                        model, data, ner_metric_meta, args.cuda, True, ner_label_mapper)
                for key, val in metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s: %s', dataset, epoch, key, val)
                path = os.path.join(args.output_dir, f'{dataset}_test_scores_{epoch}.json')
                result = {'metrics': metrics, 'predictions': predictions, 'uids': ids, 'scores': scores}
                # dump(path, result)

                path = os.path.join(args.output_dir, f'{dataset}_test_scores_{epoch}_2.json')
                # dump2(path, ids, scores, predictions)

            else:
                label_mapper = task_defs.label_mapper_map[task]
                metric_meta = task_defs.metric_meta_map[task]

                # dev
                data = dev_data_list[idx]
                with torch.no_grad():
                    metrics, predictions, scores, golds, ids = eval_model(
                        model, data, metric_meta, args.cuda, True, label_mapper)
                for key, val in metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'dev/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Dev %s:  %s', dataset, epoch, key, val)
                path = os.path.join(args.output_dir, f'{dataset}_dev_scores_{epoch}.json')
                result = {'metrics': metrics, 'predictions': predictions, 'uids': ids, 'scores': scores}
                dump(path, result)

                path = os.path.join(args.output_dir, f'{dataset}_dev_scores_{epoch}_2.json')
                dump2(path, ids, golds, predictions)

                # test
                data = test_data_list[idx]
                with torch.no_grad():
                    metrics, predictions, scores, golds, ids = eval_model(
                        model, data, metric_meta, args.cuda, True, label_mapper)
                for key, val in metrics.items():
                    if args.tensorboard:
                        tensorboard.add_scalar(f'test/{dataset}/{key}', val, global_step=epoch)
                    logger.warning('Task %s - epoch %s - Test %s: %s', dataset, epoch, key, val)
                path = os.path.join(args.output_dir, f'{dataset}_test_scores_{epoch}.json')
                result = {'metrics': metrics, 'predictions': predictions, 'uids': ids, 'scores': scores}
                dump(path, result)

                path = os.path.join(args.output_dir, f'{dataset}_test_scores_{epoch}_2.json')
                dump2(path, ids, golds, predictions)

            logger.info('[new test scores saved.]')

        if not args.not_save:
            model_file = os.path.join(args.output_dir, f'model_{epoch}.pt')
            model.save(model_file)

    if args.tensorboard:
        tensorboard.close()


if __name__ == '__main__':
    train()
