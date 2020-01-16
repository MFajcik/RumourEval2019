import re
import string
import warnings

import preprocessor as twitter_preprocessor
import spacy
# See spacy tag_map.py for tag explanation
from nltk.corpus import stopwords
from spacy.symbols import PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, VERB, NOUN, PROPN, PART, PRON, ORTH

from utils.utils import DotDict

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

nlp = None
punctuation = list(string.punctuation) + ["``"]
stopWords = set(stopwords.words('english'))

validPOS = [PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, VERB, NOUN, PROPN, PART, PRON]
POS_dict = {x: i + 2 for i, x in enumerate(validPOS)}
POS_dict['UNK'] = 0
POS_dict['EOS'] = 1

validNER = ["UNK",
            "PERSON",  # People, including fictional.
            "NORP",  # Nationalities or religious or political groups.
            "FAC",  # Buildings, airports, highways, bridges, etc.
            "ORG",  # Companies, agencies, institutions, etc.
            "GPE",  # Countries, cities, states.
            "LOC",  # Non-GPE locations, mountain ranges, bodies of water.
            "PRODUCT",  # Objects, vehicles, foods, etc. (Not services.)
            "EVENT",  # Named hurricanes, battles, wars, sports events, etc.
            "WORK_OF_ART",  # Titles of books, songs, etc.
            "LAW",  # Named documents made into laws.
            "LANGUAGE",  # Any named language.
            "DATE",  # Absolute or relative dates or periods.
            "TIME",  # Times smaller than a day.
            "PERCENT",  # Percentage, including "%".
            "MONEY",  # Monetary values, including unit.
            "QUANTITY",  # Measurements, as of weight or distance.
            "ORDINAL",  # "first", "second", etc.
            "CARDINAL",  # Numerals that do not fall under another type.
            ]

validDEPS = ['UNK',
             'acl',
             'acomp',
             'advcl',
             'advmod',
             'agent',
             'amod',
             'appos',
             'attr',
             'aux',
             'auxpass',
             'case',
             'cc',
             'ccomp',
             'complm',
             'compound',
             'conj',
             'cop',
             'csubj',
             'csubjpass',
             'dative',
             'dep',
             'det',
             'dobj',
             'expl',
             'hmod',
             'hyph',
             'infmod',
             'intj',
             'iobj',
             'mark',
             'meta',
             'neg',
             'nmod',
             'nn',
             'npadvmod',
             'nsubj',
             'nsubjpass',
             'num',
             'number',
             'nummod',
             'obj',
             'obl',
             'oprd',
             'parataxis',
             'partmod',
             'pcomp',
             'pobj',
             'poss',
             'possessive',
             'preconj',
             'predet',
             'prep',
             'prt',
             'punct',
             'quantmod',
             'rcmod',
             'relcl',
             'root',
             'xcomp']


def preprocess_text(text: str, opts, nlpengine=None, lang='en', special_tags=["<pad>", "<eos>"],
                    use_tw_preprocessor=True):
    if use_tw_preprocessor:
        ## ! There is a bug in original package for twitter preprocessing
        # Sometomes regexp for link preprocessing freezes
        # So we preprocess links separately
        text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "$URL$",
                      text.strip())
        twitter_preprocessor.set_options('mentions')
        text = twitter_preprocessor.tokenize(text)
        # processed_chunk = twitter_preprocessor.clean(text)
    if nlpengine is None:
        global nlp
        if nlp is None:
            nlp = spacy.load(lang)
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
            for x in ['URL', 'MENTION', 'HASHTAG', 'RESERVED', 'EMOJI', 'SMILEY', 'NUMBER', ]:
                nlp.tokenizer.add_special_case(f'${x}$', [{ORTH: f'${x}$'}])
        nlpengine = nlp

    BLvec = []
    POSvec = []
    DEPvec = []
    NERvec = []

    processed_chunk = ""
    doc = nlpengine(text)
    doclen = 0
    for sentence in doc.sents:
        for w in sentence:

            # Some phrases are automatically tokenized by Spacy
            # i.e. New York, in that case we want New_York in our dictionary
            word = "_".join(w.text.split())
            if word.isspace() or word == "":
                continue
            if opts.remove_stop_words and word.lower() in stopWords:
                continue

            if opts.remove_puncuation and word in punctuation:
                continue

            # Spacy lemmatized I,He/She/It into artificial
            # -PRON- lemma, which is unwanted
            if opts.lemmatize_words:
                output = w.lemma_ if w.lemma_ != '-PRON-' else w.lower_
            else:
                output = word

            if opts.to_lowercase:
                output = output.lower()

            if opts.replace_nums and output.replace('.', '', 1).isdigit():
                output = opts.num_replacement

            output = output.replace("n't", "not")
            doclen += 1
            processed_chunk += "%s " % (output)

            # Sometimes, when the word contains punctuation and we split it manually
            # the output can contain multiple tokens
            # In such case, just copy the features..., it happens rarely

            if opts.returnbiglettervector:
                BLvec.append(int(w.text[0].isupper()))
            if opts.returnposvector:
                POSvec.append(POS_dict.get(w.pos, POS_dict['UNK']))
            if opts.returnDEPvector:
                try:
                    DEPvec.append(validDEPS.index(w.dep_.lower()))
                except ValueError:
                    DEPvec.append(validDEPS.index('UNK'))
            if opts.returnNERvector:
                try:
                    NERvec.append(validNER.index(w.ent_type_))
                except ValueError:
                    NERvec.append(validNER.index('UNK'))

        if opts.add_eos:
            doclen += 1
            processed_chunk += opts.eos + "\n"
            if opts.returnbiglettervector:
                BLvec.append(0)
            if opts.returnposvector:
                POSvec.append(POS_dict['EOS'])
            if opts.returnDEPvector:
                DEPvec.append(0)
            if opts.returnNERvector:
                NERvec.append(0)
        else:
            processed_chunk += "\n"

    processed_chunk = processed_chunk.strip()
    assert len(processed_chunk.split()) == len(BLvec) == len(POSvec) == len(DEPvec) == len(NERvec)
    return processed_chunk, BLvec, POSvec, DEPvec, NERvec


def initopts():
    o = DotDict()
    o.stopwords_file = ""
    o.remove_puncuation = False
    o.remove_stop_words = False
    o.lemmatize_words = False
    o.num_replacement = "[NUM]"
    o.to_lowercase = False
    o.replace_nums = False  # Nums are important, since rumour may be lying about count
    o.eos = "[EOS]"
    o.add_eos = True
    o.returnNERvector = True
    o.returnDEPvector = True
    o.returnbiglettervector = True
    o.returnposvector = True
    return o


if __name__ == "__main__":
    print(preprocess_text(
        "Appalled by the attack on Charlie Hebdo in Paris, 10 - probably journalists - now confirmed dead. An attack on free speech everywhere.",
        initopts()))
