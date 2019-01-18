import json
from typing import List, Tuple

import torch
import torchtext as tt
from torchtext.data import Example


class RumourEval2019Dataset_Seq(tt.data.Dataset):
    empty_examples = ['d0t6et0',
                      'd0st6p1',
                      'd0t7f9d',
                      'd0st9rq',
                      'd0snspk',
                      'd0t9scw',
                      'd0t7189',
                      'd0srm52',
                      'd0t6z7v',
                      'd0t8zbx',
                      'd0texoj',
                      'd0thpei']

    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], **kwargs):
        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []

            for example in data_json["Examples"]:
                if example["tweet_id"] in RumourEval2019Dataset_Seq.empty_examples:
                    continue
                examples.append(Example.fromlist(list(example.values()), fields))
            super(RumourEval2019Dataset_Seq, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields(text_field=lambda: tt.data.Field(batch_first=True),
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
            ('spacy_processed_text', text_field()),
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
            ('spacy_processed_text_src', tt.data.RawField())
        ]
