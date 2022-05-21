"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-15
"""
from typing import Set, Dict
# 类型注释
import yaml
from data_utils.task_def import TaskType, DataFormat, EncoderModelType
from data_utils.metrics import Metric
from data_utils.vocab import Vocabulary


class MultiTaskDefs(object):
    def __init__(self, task_def_path):
        with open(task_def_path) as fa:
            self.task_def_dic = yaml.load(fa, yaml.FullLoader)
        # {'clinicalsts': {'data_format': 'PremiseAndOneHypothesis', 'encoder_type': 'BERT', ...}}
        # print(self.task_def_dic)
        self.label_mapper_map = {}  # dict[str: Vocabulary]
        self.n_class_map = {}
        self.data_format_map = {}
        self.task_type_map = {}
        self.metric_meta_map = {}
        self.enable_san_map = {}
        self.dropout_p_map = {}
        self.split_names_map = {}
        self.encoder_type = None
        # 以键值对的形式将文件读入内存
        for task, task_def in self.task_def_dic.items():
            assert "_" not in task
            # todo 简化代码
            if 'Joint-two' in task or 'Joint-bCLS-mtCLS' in task or 'Joint-mCLS-mtCLS' in task:
                self.ner_n_class_map = {}
                self.ner_n_class_map[task] = task_def['ner_nclass']
                self.cls_n_class_map = {}
                self.cls_n_class_map[task] = task_def['cls_nclass']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.ner_metric_meta_map = {}
                self.ner_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['ner_metric_meta'])
                self.cls_metric_meta_map = {}
                self.cls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['cls_metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            elif 'Joint-mt' in task:
                self.ner_n_class_map = {}
                self.ner_n_class_map[task] = task_def['ner_nclass']
                self.mt_n_class_map = {}
                self.mt_n_class_map[task] = task_def['mt_nclass']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.ner_metric_meta_map = {}
                self.ner_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['ner_metric_meta'])
                self.mt_metric_meta_map = {}
                self.mt_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['mt_metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            elif 'Joint-three-mt' in task:
                self.ner_n_class_map = {}
                self.ner_n_class_map[task] = task_def['ner_nclass']
                self.mt_n_class_map = {}
                self.mt_n_class_map[task] = task_def['mt_nclass']
                self.cls_n_class_map = {}
                self.cls_n_class_map[task] = task_def['cls_nclass']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.ner_metric_meta_map = {}
                self.ner_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['ner_metric_meta'])
                self.mt_metric_meta_map = {}
                self.mt_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['mt_metric_meta'])
                self.cls_metric_meta_map = {}
                self.cls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['cls_metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            elif 'Joint-three-cls' in task:
                self.ner_n_class_map = {}
                self.ner_n_class_map[task] = task_def['ner_nclass']
                self.bcls_n_class_map = {}
                self.bcls_n_class_map[task] = task_def['bcls_nclass']
                self.mcls_n_class_map = {}
                self.mcls_n_class_map[task] = task_def['mcls_nclass']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.ner_metric_meta_map = {}
                self.ner_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['ner_metric_meta'])
                self.bcls_metric_meta_map = {}
                self.bcls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['bcls_metric_meta'])
                self.mcls_metric_meta_map = {}
                self.mcls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['mcls_metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            elif 'Joint-all' in task:
                self.ner_n_class_map = {}
                self.ner_n_class_map[task] = task_def['ner_nclass']
                self.mt_n_class_map = {}
                self.mt_n_class_map[task] = task_def['mt_nclass']
                self.bcls_n_class_map = {}
                self.bcls_n_class_map[task] = task_def['bcls_nclass']
                self.mcls_n_class_map = {}
                self.mcls_n_class_map[task] = task_def['mcls_nclass']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.ner_metric_meta_map = {}
                self.ner_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['ner_metric_meta'])
                self.mt_metric_meta_map = {}
                self.mt_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['mt_metric_meta'])
                self.bcls_metric_meta_map = {}
                self.bcls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['bcls_metric_meta'])
                self.mcls_metric_meta_map = {}
                self.mcls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['mcls_metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            elif 'Joint-bCLS-mCLS' in task:
                self.bcls_n_class_map = {}
                self.bcls_n_class_map[task] = task_def['bcls_nclass']
                self.mcls_n_class_map = {}
                self.mcls_n_class_map[task] = task_def['mcls_nclass']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.bcls_metric_meta_map = {}
                self.bcls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['bcls_metric_meta'])
                self.mcls_metric_meta_map = {}
                self.mcls_metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['mcls_metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            else:
                self.n_class_map[task] = task_def['n_class']
                self.data_format_map[task] = DataFormat[task_def['data_format']]
                self.task_type_map[task] = TaskType[task_def['task_type']]
                self.metric_meta_map[task] = tuple(Metric[metric_name] for metric_name in task_def['metric_meta'])
                self.enable_san_map[task] = task_def['enable_san']
            # False
            if self.encoder_type is None:
                self.encoder_type = EncoderModelType[task_def['encoder_type']]
            else:
                if self.encoder_type != EncoderModelType[task_def['encoder_type']]:
                    raise ValueError('The shared Encoder should be same.')

            if 'labels' in task_def:
                label_mapper = Vocabulary(True)
                # print(label_mapper), label的词表不需要UNK这些
                for label in task_def['labels']:
                    label_mapper.add(label)
                # print(label_mapper.__len__())
                self.label_mapper_map[task] = label_mapper
            elif 'Joint-two' in task or 'Joint-bCLS-mtCLS' in task or 'Joint-mCLS-mtCLS' in task:
                ner_label_mapper = Vocabulary(True)
                for label in task_def['ner_labels']:
                    ner_label_mapper.add(label)
                cls_label_mapper = Vocabulary(True)
                for label in task_def['cls_labels']:
                    cls_label_mapper.add(label)
                self.ner_label_mapper_map = {}
                self.cls_label_mapper_map = {}
                self.ner_label_mapper_map[task] = ner_label_mapper
                self.cls_label_mapper_map[task] = cls_label_mapper
            elif 'Joint-mt' in task:
                ner_label_mapper = Vocabulary(True)
                for label in task_def['ner_labels']:
                    ner_label_mapper.add(label)
                mt_label_mapper = Vocabulary(True)
                for label in task_def['mt_labels']:
                    mt_label_mapper.add(label)
                self.ner_label_mapper_map = {}
                self.mt_label_mapper_map = {}
                self.ner_label_mapper_map[task] = ner_label_mapper
                self.mt_label_mapper_map[task] = mt_label_mapper
            elif 'Joint-three-mt' in task:
                ner_label_mapper = Vocabulary(True)
                for label in task_def['ner_labels']:
                    ner_label_mapper.add(label)
                mt_label_mapper = Vocabulary(True)
                for label in task_def['mt_labels']:
                    mt_label_mapper.add(label)
                cls_label_mapper = Vocabulary(True)
                for label in task_def['cls_labels']:
                    cls_label_mapper.add(label)
                self.ner_label_mapper_map = {}
                self.mt_label_mapper_map = {}
                self.cls_label_mapper_map = {}
                self.ner_label_mapper_map[task] = ner_label_mapper
                self.mt_label_mapper_map[task] = mt_label_mapper
                self.cls_label_mapper_map[task] = cls_label_mapper
            elif 'Joint-three-cls' in task:
                ner_label_mapper = Vocabulary(True)
                for label in task_def['ner_labels']:
                    ner_label_mapper.add(label)
                bcls_label_mapper = Vocabulary(True)
                for label in task_def['bcls_labels']:
                    bcls_label_mapper.add(label)
                mcls_label_mapper = Vocabulary(True)
                for label in task_def['mcls_labels']:
                    mcls_label_mapper.add(label)
                self.ner_label_mapper_map = {}
                self.bcls_label_mapper_map = {}
                self.mcls_label_mapper_map = {}
                self.ner_label_mapper_map[task] = ner_label_mapper
                self.bcls_label_mapper_map[task] = bcls_label_mapper
                self.mcls_label_mapper_map[task] = mcls_label_mapper
            elif 'Joint-all' in task:
                ner_label_mapper = Vocabulary(True)
                for label in task_def['ner_labels']:
                    ner_label_mapper.add(label)
                mt_label_mapper = Vocabulary(True)
                for label in task_def['mt_labels']:
                    mt_label_mapper.add(label)
                bcls_label_mapper = Vocabulary(True)
                for label in task_def['bcls_labels']:
                    bcls_label_mapper.add(label)
                mcls_label_mapper = Vocabulary(True)
                for label in task_def['mcls_labels']:
                    mcls_label_mapper.add(label)
                self.ner_label_mapper_map = {}
                self.mt_label_mapper_map = {}
                self.bcls_label_mapper_map = {}
                self.mcls_label_mapper_map = {}
                self.ner_label_mapper_map[task] = ner_label_mapper
                self.mt_label_mapper_map[task] = mt_label_mapper
                self.bcls_label_mapper_map[task] = bcls_label_mapper
                self.mcls_label_mapper_map[task] = mcls_label_mapper
            elif 'Joint-bCLS-mCLS' in task:
                bcls_label_mapper = Vocabulary(True)
                for label in task_def['bcls_labels']:
                    bcls_label_mapper.add(label)
                mcls_label_mapper = Vocabulary(True)
                for label in task_def['mcls_labels']:
                    mcls_label_mapper.add(label)
                self.bcls_label_mapper_map = {}
                self.mcls_label_mapper_map = {}
                self.bcls_label_mapper_map[task] = bcls_label_mapper
                self.mcls_label_mapper_map[task] = mcls_label_mapper
            else:
                self.label_mapper_map[task] = None

            if "dropout_p" in task_def:
                self.dropout_p_map[task] = task_def['dropout_p']

            if 'split_names' in task_def:
                self.split_names_map[task] = task_def['split_names']
            else:
                self.split_names_map[task] = ['train', 'dev', 'test']

    @property
    def task(self) -> Set[str]:
        # @property修饰属性并防止其被修改, Vocabulary.task调用
        return self.task_def_dic.keys()


if __name__ == '__main__':
    test = MultiTaskDefs('multi_task_def.yml')
    print(test.label_mapper_map['NCBI-disease-IOB'].__contains__('O'))
    print(test.label_mapper_map['NCBI-disease-IOB']['X'])
    print(test.n_class_map['NCBI-disease-IOB'])
