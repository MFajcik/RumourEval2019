import json
from collections import Counter, OrderedDict
from itertools import chain
from typing import List, Tuple

import torch
import torchtext as tt
from torchtext.data import Example
from torchtext.data.dataset import Dataset


class RumourEval2019Dataset_Branches(tt.data.Dataset):
    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], **kwargs):
        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            for e in data_json["Examples"]:
                starting_fields = [e["id"], e["stance_labels"], e["veracity_label"],
                                   self.from_list(e, "features", "spacy_processed_text")]
                examples.append(
                    Example.fromlist(starting_fields +
                                     [self.from_list(e, "features", id) for id, _ in fields[len(starting_fields):]],
                                     fields))
            super(RumourEval2019Dataset_Branches, self).__init__(examples, fields, **kwargs)

    def from_list(self, d: dict, str1: str, str2: str):
        return [d[str1][i][str2] for i in range(len(d[str1]))]

    @staticmethod
    def get_preprocess(sep_token):
        def preprocess(data: List[str]):
            return (f" {sep_token} ".join(data) + f" {sep_token} ").split()

        return preprocess

    @staticmethod
    def prepare_fields(sep_token):
        float_field = lambda: tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=-1.)
        return [
            ('id', tt.data.RawField()),
            ('stance_labels',
             tt.data.Field(use_vocab=False, batch_first=True,
                           is_target=True, pad_token=-1)),
            ('veracity_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True)),
            ('spacy_processed_text_raw', tt.data.RawField()),
            ('avgw2v', tt.data.Field(use_vocab=False, dtype=torch.float, batch_first=True, pad_token=[0.] * 300)),
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
            ('spacy_processed_text',
             tt.data.Field(batch_first=True,
                           preprocessing=RumourEval2019Dataset_Branches.get_preprocess(sep_token))),
            ('string_id', tt.data.RawField())
        ]

    @staticmethod
    def build_vocab(field, sep_token, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, _field in
                            arg.fields.items() if _field is field]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not field.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [field.unk_token, field.pad_token, field.init_token,
                            field.eos_token, sep_token]
            if tok is not None))
        field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
