"""
author:yqtong@stu.xmu.edu.cn
date:2020-10-25
"""
import tqdm
import unicodedata

PAD = 'PADPAD'
UNK = 'UNKUNK'
STA = 'BOSBOS'
# begin of sentence
END = 'EOSEOS'
# end of sentence

PAD_ID = 0
UNK_ID = 1
STA_ID = 2
END_ID = 3


class Vocabulary(object):
    INIT_LEN = 4

    def __init__(self, neat=False):
        self.neat = neat
        if not neat:
            self.tok2ind = {PAD: PAD_ID, UNK: UNK_ID, STA: STA_ID, END: END_ID}
            self.ind2tok = {PAD_ID: PAD, UNK_ID: UNK, STA_ID: STA, END_ID: END}
        else:
            self.tok2ind = {}
            self.ind2tok = {}

    def __len__(self):
        # 事先预定义好的,我们对其进行重写,就像__init__一样
        return len(self.tok2ind)

    def __iter__(self):
        # iter函数返回迭代器对象
        return iter(self.tok2ind)

    def __contains__(self, key):
        # return True or False
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return key in self.tok2ind

    def __getitem__(self, key):
        # input key return value
        if type(key) == int:
            # dict.get(key, default=None)
            # return token
            return self.ind2tok.get(key, -1) if self.neat else self.ind2tok.get(key, UNK)
        elif type(key) == str:
            # return index
            return self.tok2ind.get(key, None) if self.neat else self.tok2ind.get(key, self.tok2ind.get(UNK))

    def __setitem__(self, key, value):
        # 对Vocabulary进行修改
        if type(key) == int and type(value) == str:
            self.ind2tok[key] = value
        elif type(key) == str and type(value) == int:
            self.tok2ind[value] = key
        else:
            raise RuntimeError('Invalid (key, value) types.')

    def add(self, token):
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def get_vocab_list(self, with_order=True):
        if with_order:
            words = [self[k] for k in range(len(self))]
            # [word1, word2, ...], self[k]调用__getitem__
        else:
            words = [k for k in self.tok2ind.keys()
                     if k not in {PAD, UNK, STA, END}]
        return words

    def toidx(self, tokens):
        return [self[tok] for tok in tokens]

    def copy(self):
        new_vocab = Vocabulary(self.neat)
        for w in self:
            new_vocab.add(w)
        return new_vocab

    def build(words, neat=False):
        vocab = Vocabulary(neat)
        for w in words: vocab.add(w)
        return vocab





















