"""
author:yqtong
date:2020-12-22
"""
import logging
import os

import torch


def multi_model_strip(src_model_path, trg_model_path):
    """
    extract shared weights
    :param src_model_path:
    :param trg_model_path:
    :return:
    """
    if not os.path.exists(src_model_path):
        logging.error('%s: Cannot find the model', src_model_path)

    map_location = 'cpu' if not torch.cuda.is_available() else None
    state_dict = torch.load(src_model_path, map_location=map_location)
    config = state_dict['config']

    if config['ema_opt'] > 0:
        state = state_dict['ema']
    else:
        state = state_dict['state']
    # print(state)
    my_state = {k: v for k, v in state.items() if not k.startswith('scoring_list.')}
    my_config = {k: config[k] for k in ('vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                                        'hidden_act', 'intermediate_size', 'hidden_dropout_prob',
                                        'attention_probs_dropout_prob', 'max_position_embeddings', 'type_vocab_size',
                                        'initializer_range')}

    torch.save({'state': my_state, 'config': my_config}, trg_model_path)


if __name__ == '__main__':
    src_model_path = '18/model_80.pt'
    trg_model_path = '18/multi_task_model.pt'
    multi_model_strip(src_model_path, trg_model_path)