"""
author:yqtong@stu.xmu.edu.cn
date:2020-11-01
"""
import collections
import json


def is_whitespace(char):
    if char == ' ' or char == '\t' or char == '\r' or char == '\n' or ord(char) == 0x202F:
        # 0x202F Narrow no-break space
        return True
    return False


def parse_mrc_label(label, text):
    """
    :param label:
    :param doc:
    :return:
    """
    answer_start, answer_end, entity_type = label.split(':::')
    answer_start = eval(answer_start)
    answer_end = eval(answer_end)
    # print(answer_start)
    # print(answer_end)
    if answer_start == -1 and answer_end == -1:
        return [], [], entity_type, ''
    assert len(answer_start) == len(answer_end)
    text_list = text.split(' ')
    answer_list = []
    for idx, idy in zip(answer_start, answer_end):
        # 左闭右开, 所以+1
        answer_list.append(' '.join(text_list[idx:idy+1]))
    answer = ' '.join(answer_list)
    return answer_start, answer_end, entity_type, answer


def recompute_span(answer, answer_offset, char_to_word_offset):
    answer_length = len(answer)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]
    return start_position, end_position


def token_doc(text):
    """
    :param text:
    :return:
    """
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset


def doc_split(doc_subwords, max_tokens_for_doc, doc_stride=180):
    """
    :param doc_subwords:
    :param max_tokens_for_doc:
    :param doc_stride:
    :return:
    """
    # 有点像命名了一个DocSpan对象, 里面有个start和length属性
    _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
    doc_spans = []
    start_offset = 0
    while start_offset < len(doc_subwords):
        length = len(doc_subwords) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(doc_subwords):
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


class InputFeatures(object):
    def __init__(self,
                example_index,
                doc_span_index,
                tokens,
                token_to_orig_map,
                token_is_max_context,
                input_ids,
                input_mask,
                segment_ids,
                start_position,
                end_position,
                doc_offset=0):
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_content = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.doc_offset = doc_offset

    def __str__(self):
        # 当我们在调用print(类)时,系统会先查找__str__或者__repr__方法,如果有这两种方法的一个,则打印方法返回的值.
        return json.dumps({
            'unique_id': self.example_index,
            'example_index': self.example_index,
            'doc_span_index': self.doc_span_index,
            'tokens': self.tokens,
            'token_to_orig_map': self.token_to_orig_map,
            'token_is_max_context': self.token_is_max_content,
            'input_ids': self.input_ids,
            'input_mask': self.input_mask,
            'segment_ids': self.segment_ids,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'doc_offset': self.doc_offset
        })


def mrc_feature(tokenizer, index, query, doc_tokens, answer_start, answer_end, max_seq_length, answer_text, is_training):
    """
    :param tokenizer:
    :param index:
    :param query:
    :param doc_tokens:
    :param answer_start:
    :param answer_end:
    :param max_seq_length:
    :param answer_text:
    :param is_training:
    :return:
    """
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    query_ids = tokenizer.tokenize(query)
    # print(query_ids)
    # -3 for 2 [SEP] and 1 [CLS]
    max_tokens_for_doc = max_seq_length - len(query_ids) - 3
    for (index, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(index)
            all_doc_tokens.append(sub_token)
    # print(all_doc_tokens)
    tok_start_position = []
    tok_end_position = []

    if is_training:
        if answer_start == -1 and answer_end == -1:
            tok_start_position.append(-1)
            tok_end_position.append(-1)
        else:
            for idx, idy in zip(answer_start, answer_end):
                if idy < len(doc_tokens) - 1:
                    tok_start_position.append(orig_to_tok_index[idx])
                    tok_end_position.append(orig_to_tok_index[idy + 1] - 1)
                else:
                    tok_end_position.append(len(all_doc_tokens) - 1)
                    # todo _improve_answer_span
    # 做了sub-word以后的span
    doc_spans = doc_split(all_doc_tokens, max_tokens_for_doc=max_tokens_for_doc)
    feature_list = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = ["[CLS]"] + query_ids + ["[SEP]"]
        token_to_orig_map = {}
        token_is_max_content = {}
        segment_ids = [0 for idx in range(len(tokens))]

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_content = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_content[len(tokens)] = is_max_content
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding token.
        input_mask = [1] * len(input_ids)
        # one for CLS and one for SEP
        doc_offset = len(query_ids) + 2

        start_position = []
        end_position = []
        for idx, idy in zip(tok_start_position, tok_end_position):
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            if not (idx >= doc_start and idy <=doc_end):
                continue
            else:
                temp_start_position = idx - doc_start + doc_offset
                temp_end_position = idy - doc_start + doc_offset
                start_position.append(temp_start_position)
                end_position.append(temp_end_position)

        feature = InputFeatures(
            example_index=index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_content,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            doc_offset=doc_offset
        )
        feature_list.append(feature)
    return feature_list


if __name__ == '__main__':
    label = '[6, 12]:::[8, 16]:::disease'
    text = 'Clustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia .'
    result1, result2, result3 = parse_mrc_label(label, text)
    print(result1, result2, result3)
    doc_tokens, char_to_word_offset = token_doc(text)
    print(doc_tokens, char_to_word_offset)
    start_position, end_position = recompute_span(result3, result1, char_to_word_offset)
    print(start_position)
    print(end_position)
