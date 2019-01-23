import json
from typing import List, Tuple

import torch
import torchtext as tt
from pytorch_pretrained_bert import BertTokenizer
from torchtext.data import Example
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class RumourEval2019Dataset_BERTTriplets(tt.data.Dataset):
    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], tokenizer: BertTokenizer,
                 max_length: int = 512, include_features=False, **kwargs):
        max_length = max_length - 3  # Count without special tokens
        sentiment_analyser = SentimentIntensityAnalyzer()
        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            # Each input needs  to have at most 2 segments
            # We will create following input
            # - [CLS] source post, previous post [SEP] choice_1 [SEP]

            # more important parts are towards the end, usually, and they can be truncated

            for example in data_json["Examples"]:
                make_ids = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
                text = make_ids(example["spacy_processed_text"])
                prev = make_ids(example["spacy_processed_text_prev"])
                src = make_ids(example["spacy_processed_text_src"])
                segment_A = src + prev
                segment_B = text
                text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                           [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                # truncate if exceeds max length
                if len(text_ids) > max_length:
                    # Truncate segment A
                    segment_A = segment_A[:max_length // 2]
                    text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                               [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                    if len(text_ids) > max_length:
                        # Truncate also segment B
                        segment_B = segment_B[:max_length // 2]
                        text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                                   [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                segment_ids = [0] * (len(segment_A) + 2) + [1] * (len(segment_B) + 1)
                input_mask = [1] * len(segment_ids)
                # example_list = list(example.values())[:-3] + [text_ids, segment_ids]
                if include_features:
                    example_list = list(example.values()) + [text_ids, segment_ids,input_mask]
                else:
                    sentiment = sentiment_analyser.polarity_scores(example["raw_text"])
                    example_list = [example["id"], example["branch_id"], example["tweet_id"], example["stance_label"],
                                    example["veracity_label"],
                                    "\n-----------\n".join(
                                        [example["raw_text_src"], example["raw_text_prev"], example["raw_text"]]),
                                    example["issource"], sentiment["pos"], sentiment["neu"], sentiment["neg"]] + [
                                       text_ids, segment_ids,input_mask]

                examples.append(Example.fromlist(example_list, fields))
            super(RumourEval2019Dataset_BERTTriplets, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields_for_text():
        """
        BERT [PAD] token has index 0
        """
        text_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        return [
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text', tt.data.RawField()),
            ('issource', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_pos', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_neu', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_neg', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('text', text_field()),
            ('type_mask', text_field()),
            ('input_mask', text_field())]

    @staticmethod
    def prepare_fields_for_f_and_text(
            text_field=lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0),
            float_field=lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=False,
                                              dtype=torch.float)):
        return [
            ('avgw2v', tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True)),
            ('hasnegation', float_field()),
            ('hasswearwords', float_field()),
            ('capitalratio', float_field()),
            ('hasperiod', float_field()),
            ('hasqmark', float_field()),
            ('hasemark', float_field()),
            ('hasurl', float_field()),
            ('haspic', float_field()),
            ('charcount', float_field()),
            ('wordcount', float_field()),
            ('issource', float_field()),
            ('Word2VecSimilarityWrtOther', float_field()),
            ('Word2VecSimilarityWrtSource', float_field()),
            ('Word2VecSimilarityWrtPrev', float_field()),
            ('raw_text', tt.data.RawField()),
            ('spacy_processed_text', tt.data.RawField()),
            ('spacy_processed_BLvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_POSvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_DEPvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_NERvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('src_num_false_synonyms', float_field()),
            ('src_num_false_antonyms', float_field()),
            ('thread_num_false_synonyms', float_field()),
            ('thread_num_false_antonyms', float_field()),
            ('src_unconfirmed', float_field()),
            ('src_rumour', float_field()),
            ('thread_unconfirmed', float_field()),
            ('thread_rumour', float_field()),
            ('src_num_wh', float_field()),
            ('thread_num_wh', float_field()),
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text_prev', tt.data.RawField()),
            ('raw_text_src', tt.data.RawField()),
            ('spacy_processed_text_prev', tt.data.RawField()),
            ('spacy_processed_text_src', tt.data.RawField()),
            ('text', text_field()),
            ('type_mask', text_field()),
            ('input_mask', text_field())
        ]


class RumourEval2019Dataset_BERTTriplets_with_Tags(tt.data.Dataset):
    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], tokenizer: BertTokenizer,
                 max_length: int = 512, include_features=False, **kwargs):
        max_length = max_length - 3  # Count without special tokens
        sentiment_analyser = SentimentIntensityAnalyzer()
        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            # Each input needs  to have at most 2 segments
            # We will create following input
            # - [CLS] source post, previous post [SEP] choice_1 [SEP]

            # more important parts are towards the end, usually, and they can be truncated

            for example in data_json["Examples"]:
                def make_ids_with_mapping(text):

                    make_ids = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
                    word_tokens = text.split()
                    ids_mapping = {i: make_ids(w) for i, w in enumerate(word_tokens)}
                    ids = []
                    for subwordindices in ids_mapping.values(): ids+=subwordindices
                    return ids_mapping, ids

                text_mapping, text = make_ids_with_mapping(example["spacy_processed_text"])
                prev_mapping, prev = make_ids_with_mapping(example["spacy_processed_text_prev"])
                src_mapping, src = make_ids_with_mapping(example["spacy_processed_text_src"])

                segment_A = src + prev
                segment_B = text
                text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                           [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                NER_segment_B = []
                for k,v in text_mapping.items():
                    for id in v: NER_segment_B.append(example['spacy_processed_NERvec'][k])

                DEP_segment_B = []
                for k, v in text_mapping.items():
                    for id in v: DEP_segment_B.append(example['spacy_processed_DEPvec'][k])

                POS_segment_B = []
                for k, v in text_mapping.items():
                    for id in v: POS_segment_B.append(example['spacy_processed_POSvec'][k])

                # truncate if exceeds max length
                if len(text_ids) > max_length:
                    # Truncate segment A
                    segment_A = segment_A[:max_length // 2]
                    text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                               [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                    if len(text_ids) > max_length:
                        # Truncate also segment B
                        segment_B = segment_B[:max_length // 2]
                        text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + \
                                   [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                        NER_segment_B = NER_segment_B[:max_length // 2]
                        DEP_segment_B = DEP_segment_B[:max_length // 2]
                        POS_segment_B = POS_segment_B[:max_length // 2]

                NER_mask_ids = [0] * (len(segment_A) + 2)+NER_segment_B+[0]
                DEP_mask_ids = [0] * (len(segment_A) + 2)+DEP_segment_B+[0]
                POS_mask_ids = [0] * (len(segment_A) + 2)+POS_segment_B+[0]
                segment_ids = [0] * (len(segment_A) + 2) + [1] * (len(segment_B) + 1)
                # example_list = list(example.values())[:-3] + [text_ids, segment_ids]
                if include_features:
                    example_list = list(example.values()) + [text_ids, segment_ids,NER_mask_ids]
                else:
                    sentiment = sentiment_analyser.polarity_scores(example["raw_text"])
                    example_list = [example["id"], example["branch_id"], example["tweet_id"], example["stance_label"],
                                    example["veracity_label"],
                                    "\n-----------\n".join(
                                        [example["raw_text_src"], example["raw_text_prev"], example["raw_text"]]),
                                    example["issource"], sentiment["pos"], sentiment["neu"], sentiment["neg"]] + [
                                       text_ids, segment_ids,NER_mask_ids,DEP_mask_ids,POS_mask_ids]

                examples.append(Example.fromlist(example_list, fields))
            super(RumourEval2019Dataset_BERTTriplets_with_Tags, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields_for_text():
        """
        BERT [PAD] token has index 0
        """
        text_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        return [
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text', tt.data.RawField()),
            ('issource', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_pos', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_neu', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('sentiment_neg', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('text', text_field()),
            ('type_mask', text_field()),
            ('ner_mask', text_field()),
            ('dep_mask', text_field()),
            ('pos_mask', text_field())]

    @staticmethod
    def prepare_fields_for_f_and_text(
            text_field=lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0),
            float_field=lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=False,
                                              dtype=torch.float)):
        return [
            ('avgw2v', tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True)),
            ('hasnegation', float_field()),
            ('hasswearwords', float_field()),
            ('capitalratio', float_field()),
            ('hasperiod', float_field()),
            ('hasqmark', float_field()),
            ('hasemark', float_field()),
            ('hasurl', float_field()),
            ('haspic', float_field()),
            ('charcount', float_field()),
            ('wordcount', float_field()),
            ('issource', float_field()),
            ('Word2VecSimilarityWrtOther', float_field()),
            ('Word2VecSimilarityWrtSource', float_field()),
            ('Word2VecSimilarityWrtPrev', float_field()),
            ('raw_text', tt.data.RawField()),
            ('spacy_processed_text', tt.data.RawField()),
            ('spacy_processed_BLvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_POSvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_DEPvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_NERvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('src_num_false_synonyms', float_field()),
            ('src_num_false_antonyms', float_field()),
            ('thread_num_false_synonyms', float_field()),
            ('thread_num_false_antonyms', float_field()),
            ('src_unconfirmed', float_field()),
            ('src_rumour', float_field()),
            ('thread_unconfirmed', float_field()),
            ('thread_rumour', float_field()),
            ('src_num_wh', float_field()),
            ('thread_num_wh', float_field()),
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text_prev', tt.data.RawField()),
            ('raw_text_src', tt.data.RawField()),
            ('spacy_processed_text_prev', tt.data.RawField()),
            ('spacy_processed_text_src', tt.data.RawField()),
            ('text', text_field()),
            ('type_mask', text_field()),
            ('ner_mask', text_field())
        ]


# Did not seem to make difference
class RumourEval2019Dataset_BERTTriplets_3Segments(tt.data.Dataset):
    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], tokenizer: BertTokenizer,
                 max_length: int = 512, include_features=False, **kwargs):
        max_length = max_length - 3  # Count without special tokens

        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            # Each input needs  to have at most 2 segments
            # We will create following input
            # - [CLS] source post, previous post [SEP] choice_1 [SEP]

            # TODO: try reverting order of sentences
            # more important parts are towards the end, usually, and they can be truncated

            for example in data_json["Examples"]:
                make_ids = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
                text = make_ids(example["spacy_processed_text"])
                prev = make_ids(example["spacy_processed_text_prev"])
                src = make_ids(example["spacy_processed_text_src"])
                segment_A = src
                segment_C = prev
                segment_B = text
                text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + [tokenizer.vocab["[SEP]"]] + segment_C + \
                           [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                # truncate if exceeds max length
                if len(text_ids) > max_length:
                    # Truncate segment A
                    segment_C = segment_C[:max_length // 2]
                    text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + [tokenizer.vocab["[SEP]"]] + segment_C + \
                               [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                    if len(text_ids) > max_length:
                        # Truncate segment A
                        segment_A = segment_A[:max_length // 2]
                        text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + [tokenizer.vocab["[SEP]"]] + segment_C + \
                                   [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]
                        if len(text_ids) > max_length:
                            # Truncate also segment B
                            segment_B = segment_B[:max_length // 2]
                            text_ids = [tokenizer.vocab["[CLS]"]] + segment_A + [tokenizer.vocab["[SEP]"]] + segment_C + \
                                       [tokenizer.vocab["[SEP]"]] + segment_B + [tokenizer.vocab["[SEP]"]]

                segment_ids = [0] * (len(segment_A) + 2) + [2] * (len(segment_C) + 1) + [1] * (len(segment_B) + 1)
                # example_list = list(example.values())[:-3] + [text_ids, segment_ids]
                if include_features:
                    example_list = list(example.values()) + [text_ids, segment_ids]
                else:
                    example_list = [example["id"], example["branch_id"], example["tweet_id"], example["stance_label"],
                                    example["veracity_label"],
                                    "\n-----------\n".join(
                                        [example["raw_text_src"], example["raw_text_prev"], example["raw_text"]]),
                                    example["issource"]] + [text_ids, segment_ids]

                examples.append(Example.fromlist(example_list, fields))
            super(RumourEval2019Dataset_BERTTriplets_3Segments, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields_for_text():
        """
        BERT [PAD] token has index 0
        """
        text_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        return [
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text', tt.data.RawField()),
            ('issource', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('text', text_field()),
            ('type_mask', text_field())]

    @staticmethod
    def prepare_fields_for_f_and_text(
            text_field=lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0),
            float_field=lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=False,
                                              dtype=torch.float)):
        return [
            ('avgw2v', tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True)),
            ('hasnegation', float_field()),
            ('hasswearwords', float_field()),
            ('capitalratio', float_field()),
            ('hasperiod', float_field()),
            ('hasqmark', float_field()),
            ('hasemark', float_field()),
            ('hasurl', float_field()),
            ('haspic', float_field()),
            ('charcount', float_field()),
            ('wordcount', float_field()),
            ('issource', float_field()),
            ('Word2VecSimilarityWrtOther', float_field()),
            ('Word2VecSimilarityWrtSource', float_field()),
            ('Word2VecSimilarityWrtPrev', float_field()),
            ('raw_text', tt.data.RawField()),
            ('spacy_processed_text', tt.data.RawField()),
            ('spacy_processed_BLvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_POSvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_DEPvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('spacy_processed_NERvec',
             tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=0)),
            ('src_num_false_synonyms', float_field()),
            ('src_num_false_antonyms', float_field()),
            ('thread_num_false_synonyms', float_field()),
            ('thread_num_false_antonyms', float_field()),
            ('src_unconfirmed', float_field()),
            ('src_rumour', float_field()),
            ('thread_unconfirmed', float_field()),
            ('thread_rumour', float_field()),
            ('src_num_wh', float_field()),
            ('thread_num_wh', float_field()),
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text_prev', tt.data.RawField()),
            ('raw_text_src', tt.data.RawField()),
            ('spacy_processed_text_prev', tt.data.RawField()),
            ('spacy_processed_text_src', tt.data.RawField()),
            ('text', text_field()),
            ('type_mask', text_field())
        ]
