# encoding: utf-8


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torch.nn.modules import BCEWithLogitsLoss


def mrc_loss_compute(start_logits, end_logits, span_logits, start_labels, end_labels, match_label,
                     start_label_mask, end_label_mask, span_loss_candidates, loss_type):
    """
    :param start_logits:
    :param end_logits:
    :param span_logits:
    :param start_labels:
    :param end_labels:
    :param match_label:
    :param start_label_mask:
    :param end_label_mask:
    :param span_loss_candidates:
    :param loss_type:
    :return:
    """
    batch_size, seq_len = start_logits.size()
    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    if span_loss_candidates == "all":
        # naive mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    else:
        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if span_loss_candidates == "gold":
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )

        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    if loss_type == "bce":
        bce_loss = BCEWithLogitsLoss(reduction='none')
        start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = bce_loss(span_logits.view(batch_size, -1), match_label.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
    else:
        dice_loss = DiceLoss(with_logits=True, smooth=1e-8)
        start_loss = dice_loss(start_logits, start_labels.float(), start_float_label_mask)
        end_loss = dice_loss(end_logits, end_labels.float(), end_float_label_mask)
        match_loss = dice_loss(span_logits, match_label.float(), float_match_label_mask)

    return start_loss, end_loss, match_loss

class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)

    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
            True: the loss combines a `sigmoid` layer and the `BCELoss` in one single class.
            False: the loss contains `BCELoss`.
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss()
        >>> input = torch.randn(3, 1, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean") -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        flat_input = input.view(-1)
        flat_target = target.view(-1)

        if self.with_logits:
            flat_input = torch.sigmoid(flat_input)

        if mask is not None:
            mask = mask.view(-1).float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask

        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            return 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            return 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input,), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}"
