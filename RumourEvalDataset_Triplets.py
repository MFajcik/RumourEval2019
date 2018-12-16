import json
from collections import Counter, OrderedDict
from itertools import chain
from typing import List, Tuple

import torchtext as tt
from pytorch_pretrained_bert import BertTokenizer
from torchtext.data import Example
from torchtext.data.dataset import Dataset


class RumourEval2019Dataset_BERTTriplets(tt.data.Dataset):
    def __init__(self, path: str, fields: List[Tuple[str, tt.data.Field]], tokenizer: BertTokenizer,
                 max_length: int = 512, **kwargs):
        max_length = max_length - 3  # Count without special tokens
        with open(path) as dataf:
            data_json = json.load(dataf)
            examples = []
            # Each input needs  to have at most 2 segments
            # We will create following input
            # - [CLS] source post, previous post [SEP] choice_1 [SEP]

            # TODO: try also different configurations
            # - [CLS] source post [SEP] target_post [SEP]
            # - [CLS] source post, previous post [SEP] target_post [SEP]
            # - [CLS] source post [SEP] previous post [SEP] target_post [SEP]

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
                example_list = list(example.values())[:-3] + [text_ids, segment_ids]

                examples.append(Example.fromlist(example_list, fields))
            super(RumourEval2019Dataset_BERTTriplets, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def prepare_fields():
        """
        BERT [PAD] token has index 0
        """
        text_field = lambda: tt.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        return [
            ('id', tt.data.RawField()),
            ('branch_id', tt.data.RawField()),
            ('stance_label', tt.data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('raw_text', tt.data.RawField()),
            ('issource', tt.data.Field(use_vocab=False, batch_first=True, sequential=False)),
            ('text', text_field()),
            ('type_mask', text_field())]

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
