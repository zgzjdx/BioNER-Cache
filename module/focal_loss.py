from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.elipson = 0.000001

    def forward(self, logits, labels, probability=False):
        """
        1、概率二维、标签一维，如文本分类任务(情感分类)
        logits=(B,C),labels=(B)
        2、概率三维、标签二维，如序列标注任务(分词)
        logits=(B,T,C),labels=(B,T)
        3、概率二维、标签二维，如文本多标签任务(新闻属性)
        logits=(B,C_),labels=(B,C_)
        4、概率三维、标签三维，如序列多标签任务(事件要素抽取)
        logits=(B,T,C_),labels=(B,T,C_)
        其中，1和2等价，3和4等价

        :param logits:
        :param labels:
        :param method:
        :return:
        """
        device = logits.device
        logits_dim = logits.dim()
        labels_dim = labels.dim()

        # 3、4的多标签任务需要转为1、2的二分类任务
        if logits_dim == labels_dim:
            # logits=>(B,C_,1) or (B,T,C_,1)
            logits = logits.unsqueeze(-1)

        # 二分类任务(C=1)，通过sigmoid计算概率，再转为[1-p,p]
        if logits.size(-1) == 1:
            # 转为概率
            if not probability:
                logits_1 = torch.sigmoid(logits)
            logits_0 = 1 - logits_1
            logits_p = torch.cat([logits_0, logits_1], dim=-1)
        # 多分类任务(C>1)，通过softmax计算概率
        else:
            # 转为概率
            if not probability:
                logits_p = torch.softmax(logits, dim=-1)
            else:
                logits_p = logits

        C = logits_p.size(-1)

        if logits_dim > labels_dim:
            # 三维转为二维，logits=(B,T,C)=>(B*T,C),labels=(B,T)=>(B*T,)
            if labels_dim == 2:
                B, T, _ = logits_p.size()
                logits_p = logits_p.contiguous().view(B * T, C)
                labels = labels.contiguous().view(B * T)
        else:
            # 三维转为二维，logits=(B,C_,2)=>(B*C_,2),labels=(B,C_)=>(B*C_,)
            if labels_dim == 2:
                B, C_, _ = logits_p.size()
                logits_p = logits_p.contiguous().view(B * C_, C)
                labels = labels.contiguous().view(B * C_)
            # 四维转为二维，logits=(B,T,C_,2)=>(B*T*C_,2),labels=(B,T,C_)=>(B*T*C_,)
            else:
                B, T, C_, _ = logits_p.size()
                logits_p = logits_p.contiguous().view(B * T * C_, C)
                labels = labels.contiguous().view(B * T * C_)

        label_onehot = torch.zeros_like(logits_p).to(device).scatter_(1, labels.unsqueeze(-1), 1)
        pt_log = label_onehot * torch.log(logits_p)
        sub_pt = 1 - logits_p
        fl = -self.alpha * (sub_pt) ** self.gamma * pt_log
        fl = fl.sum(dim=-1)

        # 还原为timestep的形式用于后续对loss做mask
        if logits_dim > labels_dim:
            # 1、概率二维、标签一维，如文本分类任务(情感分类)
            if labels_dim == 1:
                fl = fl.mean()
            # 2、概率三维、标签二维，如序列标注任务(分词)
            else:
                fl = fl.view([B, T])
        else:
            # 3、概率二维、标签二维，如文本多标签任务(新闻属性)
            if labels_dim == 2:
                fl = fl.mean()
            # 4、概率三维、标签三维，如序列多标签任务(事件要素抽取)
            else:
                fl = fl.view([B, T, -1])
                fl = fl.mean(dim=-1)

        return fl