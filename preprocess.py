"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-15
"""
import os
from data_utils.logger_wrapper import create_logger
from multi_exp_def import MultiTaskDefs
from multi_utils import *


def data_process(data_path, data_name):
    """
    :param data_path:
    :param data_name:
    :return:
    """
    root = os.getcwd()
    log_file = os.path.join(root, 'data_prepro.log')
    # 在直接运行本脚本时, __name__ == "__main__"
    logger = create_logger(__name__, to_disk=True, log_file=log_file)
    task_defs = MultiTaskDefs(data_path)

    canonical_data_suffix = 'canonical_data'
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    if data_name == 'all':
        tasks = task_defs.task
    else:
        tasks = data_name.split(',')
    for task in tasks:
        logger.info("Current task %s" % task)
        data_format = task_defs.data_format_map[task]
        # 数据集名称
        split_names = task_defs.split_names_map[task]

        if task not in task_defs.task_def_dic:
            raise KeyError('%s: Cannot process this task' % task)
        # todo 优化
        if task in ['clinicalsts', 'biosses']:
            # sentence similarity
            load = load_sts
        elif task == 'mednli':
            load = load_mednli
        elif task in ('chemprot', 'i2b2-2010-re', 'ddi2013-type'):
            load = load_relation
        elif task in ('bc5cdr-disease', 'bc5cdr-chemical', 'shareclefe', 'BioNLP13CG-IOB'):
            load = load_ner
        elif task in ('NCBI-disease-IOB', 'bakeoff-2006-IOB', 'JNLPBA-IOB', 'BC2GM-IOB', 'BC5CDR-chem-IOB'):
            load = load_ner
        elif task in ('NCBI-disease-IOB-half', 'BC5CDR-chem-IOB-half', 'BC2GM-IOB-half',
                      'NCBI-disease-IOB-quarter', 'BC5CDR-chem-IOB-quarter', 'BC2GM-IOB-quarter',
                      'NCBI-disease-IOB-deci', 'BC5CDR-chem-IOB-deci', 'BC2GM-IOB-deci'):
            load = load_ner
        elif task in ('NCBI-disease-mtCLS', 'BC2GM-mtCLS', 'BC5CDR-chem-mtCLS'):
            load = load_ner
        elif task in ('NLM-BC5CDR-chem-IOB'):
            load = load_ner
        elif task in ('NCBI-disease-bCLS', 'BC2GM-bCLS', 'BC5CDR-chem-bCLS'):
            load = load_cls
        elif task in ('NCBI-disease-3CLS', 'NCBI-disease-4CLS', 'NCBI-disease-5CLS', 'NCBI-disease-mCLS'):
            load = load_cls
        elif task in ('LUO-3CLS'):
            load = load_cls
        elif task in ('BC5CDR-chem-3CLS', 'BC5CDR-chem-4CLS', 'BC5CDR-chem-5CLS', 'BC5CDR-chem-mCLS'):
            load = load_cls
        elif task in ('BC2GM-3CLS', 'BC2GM-4CLS', 'BC2GM-5CLS', 'BC2GM-mCLS'):
            load = load_cls
        elif task in ('NCBI-disease-MRC-rule', 'NCBI-disease-MRC-wiki', 'BC5CDR-chem-MRC-rule', 'BC5CDR-chem-MRC-wiki',
                      'BC2GM-MRC-rule', 'BC2GM-MRC-wiki', 'NCBI-disease-MRC-ours-three', 'BC5CDR-chem-MRC-ours-three',
                      'BC2GM-MRC-ours-three', 'NCBI-disease-MRC-none', 'BC2GM-MRC-none', 'BC5CDR-chem-MRC-none',
                      'NCBI-disease-MRC-ours-none', 'BC2GM-MRC-ours-none', 'BC5CDR-chem-MRC-ours-none',
                      'NCBI-disease-MRC-ours-one', 'BC2GM-MRC-ours-one', 'BC5CDR-chem-MRC-ours-one',
                      'NCBI-disease-MRC-ours-two', 'BC2GM-MRC-ours-two', 'BC5CDR-chem-MRC-ours-two',
                      'NCBI-disease-MRC-ours-four', 'BC2GM-MRC-ours-four', 'BC5CDR-chem-MRC-ours-four',
                      'NCBI-disease-MRC-ours-five', 'BC2GM-MRC-ours-five', 'BC5CDR-chem-MRC-ours-five',
                      'NCBI-disease-MRC-ours-random', 'BC2GM-MRC-ours-random', 'BC5CDR-chem-MRC-ours-random',
                      'BC5CDR-chem-MRC-wiki-half', 'BC5CDR-chem-MRC-wiki-quarter', 'BC5CDR-chem-MRC-wiki-deci',
                      'BC2GM-MRC-rule-half', 'BC2GM-MRC-rule-quarter', 'BC2GM-MRC-rule-deci',
                      'BC5CDR-chem-MRC-wiki-double', 'BC2GM-MRC-rule-double', 'NCBI-disease-MRC-rule-double',
                      'BC5CDR-chem-MRC-wiki-triple', 'BC2GM-MRC-rule-triple',
                      'NCBI-disease-MRC-rule-half', 'NCBI-disease-MRC-rule-quarter', 'NCBI-disease-MRC-rule-deci'):
            assert data_format == DataFormat.OneMRCAndOneSequence
            load = load_mrc_joint
        elif task in ('BC2GM-MRC-deci', 'BC5CDR-chem-MRC-deci'):
            load = load_mrc
        elif task in ('BC5CDR-chem-Joint-two-bCLS-deci', 'BC5CDR-chem-Joint-two-bCLS-quarter', 'BC5CDR-chem-Joint-two-bCLS',
                      'BC5CDR-chem-Joint-two-bCLS-half', 'NCBI-disease-Joint-two-bCLS-deci', 'NCBI-disease-Joint-two-bCLS',
                      'NCBI-disease-Joint-two-bCLS-half', 'NCBI-disease-Joint-two-bCLS-quarter', 'BC2GM-Joint-two-bCLS',
                      'BC2GM-Joint-two-bCLS-quarter', 'BC2GM-Joint-two-bCLS-half', 'BC2GM-Joint-two-bCLS-deci'
                      ):
            if data_format == DataFormat.OnePremiseAndOneSequence:
                load = load_one_premise_and_one_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('BC5CDR-chem-Joint-two-mCLS-deci', 'BC5CDR-chem-Joint-two-mCLS', 'BC2GM-Joint-two-mCLS',
                      'NCBI-disease-Joint-two-mCLS'):
            if data_format == DataFormat.OnePremiseAndOneSequence:
                load = load_one_premise_and_one_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('BC2GM-Joint-bCLS-mtCLS', 'BC5CDR-chem-Joint-bCLS-mtCLS', 'NCBI-disease-Joint-bCLS-mtCLS',
                      'BC2GM-Joint-mCLS-mtCLS', 'BC5CDR-chem-Joint-mCLS-mtCLS', 'NCBI-disease-Joint-mCLS-mtCLS'):
            if data_format == DataFormat.OnePremiseAndOneSequence:
                load = load_one_premise_and_one_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ("BC5CDR-chem-Joint-mt", 'NCBI-disease-Joint-mt', 'BC2GM-Joint-mt'):
            if data_format == DataFormat.OneSequenceAndOneSequence:
                load = load_one_sequence_and_one_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('BC5CDR-chem-Joint-three-mt-deci', 'BC5CDR-chem-Joint-three-mt', 'BC5CDR-chem-Joint-three-mt-mCLS'):
            if data_format == DataFormat.OnePremiseAndTwoSequence:
                load = load_one_premise_and_two_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('BC5CDR-chem-Joint-three-cls'):
            if data_format == DataFormat.TwoPremiseAndOneSequence:
                load = load_two_premise_and_one_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('BC5CDR-chem-Joint-all-cls', 'NCBI-disease-Joint-all-cls', 'BC2GM-Joint-all-cls',
                      'BC5CDR-chem-Joint-all-cls-half', 'BC5CDR-chem-Joint-all-cls-quarter', 'BC5CDR-chem-Joint-all-cls-deci',
                      'BC2GM-Joint-all-cls-half', 'BC2GM-Joint-all-cls-quarter', 'BC2GM-Joint-all-cls-deci',
                      'NCBI-disease-Joint-all-cls-half', 'NCBI-disease-Joint-all-cls-quarter', 'NCBI-disease-Joint-all-cls-deci'):
            if data_format == DataFormat.TwoPremiseAndTwoSequence:
                load = load_two_premise_and_two_sequence
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('BC2GM-Joint-bCLS-mCLS', 'BC5CDR-chem-Joint-bCLS-mCLS', 'NCBI-disease-Joint-bCLS-mCLS'):
            if data_format == DataFormat.TwoPremise:
                load = load_two_premise
            else:
                raise KeyError('%s: Cannot process this task' % task)
        elif task in ('NCBI-disease-Joint-all-rule', 'BC2GM-Joint-all-rule', 'BC5CDR-chem-Joint-all-rule'):
            if data_format == DataFormat.TwoPremiseAndTwoSequenceAndMRC:
                load = load_TwoPremiseAndTwoSequenceAndMRC
            else:
                raise KeyError('%s: Cannot process this task' % task)
        else:
            raise KeyError('%s: Cannot process this task' % task)

        for split_name in split_names:
            if 'MRC' in task:
                fin = os.path.join(root, f'dataset/MRC/{task}/{split_name}.json')
            elif 'Joint-bCLS-mCLS' in task:
                fin = os.path.join(root, f'dataset/Joint-bCLS-mCLS/{task}/{split_name}.tsv')
            elif 'Joint-bCLS-mtCLS' in task:
                fin = os.path.join(root, f'dataset/Joint-bCLS-mtCLS/{task}/{split_name}.tsv')
            elif 'Joint-mCLS-mtCLS' in task:
                fin = os.path.join(root, f'dataset/Joint-mCLS-mtCLS/{task}/{split_name}.tsv')
            elif 'Joint-two' in task:
                fin = os.path.join(root, f'dataset/Joint-two/{task}/{split_name}.tsv')
            elif 'Joint-mt' in task:
                fin = os.path.join(root, f'dataset/Joint-mt/{task}/{split_name}.tsv')
            elif 'Joint-three-mt' in task:
                fin = os.path.join(root, f'dataset/Joint-three-mt/{task}/{split_name}.tsv')
            elif 'Joint-three-cls' in task:
                fin = os.path.join(root, f'dataset/Joint-three-cls/{task}/{split_name}.tsv')
            elif 'Joint-all-cls' in task:
                fin = os.path.join(root, f'dataset/Joint-all-cls/{task}/{split_name}.tsv')
            elif 'Joint-all' in task:
                fin = os.path.join(root, f'dataset/Joint-all/{task}/{split_name}.json')
            elif 'IOB' in task:
                fin = os.path.join(root, f'dataset/NER/{task}/{split_name}.tsv')
            elif 'CLS' in task:
                fin = os.path.join(root, f'dataset/CLS/{task}/{split_name}.tsv')
            else:
                raise ValueError
            fout = os.path.join(canonical_data_root, f'{task}_{split_name}.tsv')
            if os.path.exists(fout):
                logger.warning('write path %s exists!' % fout)
            # load source-side data
            data = load(fin)
            logger.info('%s: Loaded %s %s samples', task, len(data), split_name)
            # write target-side data
            dump_rows(data, fout, data_format)


if __name__ == '__main__':
    data_path = 'multi_task_def.yml'
    data_name = 'all'
    data_process(data_path, data_name)
