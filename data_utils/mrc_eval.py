"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-01
"""
import string
import re
from collections import Counter


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace
    :param s:
    :return:
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    :param prediction:
    :param ground_truth:
    :return:
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    :param prediction:
    :param ground_truth:
    :return:
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    """
    :param metric_fn:
    :param predictions:
    :param ground_truths: 可能有多个参考答案
    :return:
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(predictions, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_func(human, predictions):
    """
    :param human:
    :param predictions:
    :return:
    """
    f1 = exact_match = total = 0
    for uid, ground_truths in human.items():
        total += 1
        if uid not in predictions:
            message = 'Unanswered question ' + uid + ' will receive score 0.'
            print(message)
            continue
        prediction = predictions[uid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths
        )

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return str({
        'exact_match': exact_match,
        'f1': f1
    })
