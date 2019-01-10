import json
from typing import List, Tuple

import torch
import torchtext as tt
from pytorch_pretrained_bert import BertTokenizer
from torchtext.data import Example


class RumourEval2019Dataset_BERTTriplets(tt.data.Dataset):
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
                # example_list = list(example.values())[:-3] + [text_ids, segment_ids]
                if include_features:
                    example_list = list(example.values()) + [text_ids, segment_ids]
                else:
                    example_list = [example["id"], example["branch_id"], example["tweet_id"], example["stance_label"],
                                    "\n-----------\n".join(
                                        [example["raw_text_src"], example["raw_text_prev"], example["raw_text"]]),
                                    example["issource"]] + [text_ids, segment_ids]

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
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('tweet_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text_prev', tt.data.RawField()),
            ('raw_text_src', tt.data.RawField()),
            ('spacy_processed_text_prev', tt.data.RawField()),
            ('spacy_processed_text_src', tt.data.RawField()),
            ('text', text_field()),
            ('type_mask', text_field())
        ]
