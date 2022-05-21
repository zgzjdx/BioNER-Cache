"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-15
"""

from enum import IntEnum
# 枚举类型


class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    Span = 4
    SequenceLabeling = 5
    ReadingComprehension = 6
    Joint = 7
    Joint_mt = 8
    Joint_mt_three = 9
    Joint_cls_three = 10
    Joint_all_cls = 11
    Joint_bCLS_mCLS = 12
    Joint_mrc_ner = 13
    Joint_all = 14


class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    Sequence = 4
    MRC = 5
    OnePremiseAndOneSequence = 6
    OneSequenceAndOneSequence = 7
    OnePremiseAndTwoSequence = 8
    TwoPremiseAndOneSequence = 9
    TwoPremiseAndTwoSequence = 10
    TwoPremise = 11
    OneMRCAndOneSequence = 12
    TwoPremiseAndTwoSequenceAndMRC = 13


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3