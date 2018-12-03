import string
import warnings

import spacy
# See spacy tag_map.py for tag explanation
from nltk.corpus import stopwords
from spacy.symbols import PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, VERB, NOUN, PROPN, PART, PRON

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

nlp = None
punctuation = list(string.punctuation) + ["``"]
stopWords = set(stopwords.words('english'))
validPOS = [PUNCT, SYM, ADJ, CCONJ, NUM, DET, ADV, ADP, VERB, NOUN, PROPN, PART, PRON]
POS_dict = {x: i + 2 for i, x in enumerate(validPOS)}
POS_dict['UNK'] = 0
POS_dict['EOS'] = 1


def preprocess_text(text: str, opts, nlpengine=None, lang='en', special_tags=["<pad>", "<eos>"]):
    if nlpengine is None:
        global nlp
        if nlp is None:
            nlp = spacy.load(lang)
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlpengine = nlp
    if opts.returnbiglettervector:
        BLvec = []
    if opts.returnposvector:
        POSvec = []

    processed_chunk = ""
    doc = nlpengine(text, disable=['ner', 'parser'])
    lp = None
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
            if opts.postag_words and opts.lemmatize_words:
                raw_lemma = w.lemma_ if w.lemma_ != '-PRON-' else w.lower_
                word_lemma = "_".join(raw_lemma.split())
                output = "{}_{}".format(word_lemma, w.pos_)
            elif opts.postag_words:
                output = "{}_{}".format(word, w.pos_)
            elif opts.lemmatize_words:
                raw_lemma = w.lemma_ if w.lemma_ != '-PRON-' else w.lower_
                word_lemma = "_".join(raw_lemma.split())
                output = word_lemma
            else:
                output = word

            if opts.to_lowercase and not opts.lemmatize_words:
                output = output.lower()
            if opts.replace_nums and output.replace('.', '', 1).isdigit():
                output = opts.num_replacement
            # Remove rest of word behind backslash
            if opts.remove_backslash_text:
                output = output.split("\\", maxsplit=2)[0]

            # Spacy does not tokenize /,& for some reason
            fix_puncuation = ['/', '&']
            for p in fix_puncuation:
                output = f' {p} '.join(output.split(p)).strip()

            if opts.replace_special_tags and output in special_tags:
                output = opts.special_tag_replacement

            processed_chunk += "%s " % (output)

            # Sometimes, when the word contains punctuation and we split it manually
            # the output can contain multiple tokens
            # In such case, just copy the features..., it happens rarely

            if opts.returnbiglettervector:
                outplen = len(output.split())
                for x in range(outplen):
                    BLvec.append(int(w.text[0].isupper()))
            if opts.returnposvector:
                outplen = len(output.split())
                for x in range(outplen):
                    POSvec.append(POS_dict.get(w.pos, POS_dict['UNK']))

        if opts.add_eos:
            processed_chunk += opts.eos + "\n"
            if opts.returnbiglettervector:
                BLvec.append(0)
            if opts.returnposvector:
                POSvec.append(POS_dict['EOS'])
        else:
            processed_chunk += "\n"

    if opts.returnbiglettervector and opts.returnposvector:
        return processed_chunk.strip(), BLvec, POSvec

    if opts.returnbiglettervector:
        return processed_chunk.strip(), BLvec
    if opts.returnposvector:
        return processed_chunk.strip(), POSvec

    return processed_chunk.strip()
