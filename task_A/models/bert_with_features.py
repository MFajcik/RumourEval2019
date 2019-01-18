import logging
import math

import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from torch import nn
import torch.nn.functional as F


def dump_batch_contents(batch):
    logging.debug("#" * 30)
    logging.debug("Dumping batch contents")
    for i in range(batch.text.shape[0]):
        logging.debug(f"L:{len(batch.text[i])} T: {batch.raw_text[i]}")


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class BertModelForStanceClassificationWFeatures(PreTrainedBertModel):
    """
        `input_ids`: a torch.LongTensor of shape [batch_siz, sequence_length]
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).

        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
    """

    def __init__(self, config, classes=4):
        super(BertModelForStanceClassificationWFeatures, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_layer = nn.Linear(config.hidden_size*2 + 0, config.hidden_size*2+20 + 0)
        self.last_layer = nn.Linear(config.hidden_size*2+20, classes)
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        _, pooled_output = self.bert(batch.text, batch.type_mask, None, output_all_encoded_layers=False)
        used_features = [
            batch.hasnegation,
            batch.hasswearwords,
            # batch.capitalratio,
            # batch.hasperiod,
            # batch.hasqmark,
            # batch.hasemark,
            # batch.hasurl,
            # batch.haspic,
            # batch.charcount,
            # batch.wordcount,
            batch.src_num_false_synonyms,
            batch.src_num_false_antonyms,
            batch.src_unconfirmed,
            batch.src_rumour,
            batch.src_num_wh,
            batch.issource,
            # batch.Word2VecSimilarityWrtOther,
            # batch.Word2VecSimilarityWrtSource,
            # batch.Word2VecSimilarityWrtPrev
        ]
        features = torch.cat(
            tuple([f.unsqueeze(-1) for f in used_features]), dim=-1)

        # Dropout bert output
        pooled_output = self.dropout(torch.cat((pooled_output,features),1))

        # add features
        #pooled_output_with_src = torch.cat((pooled_output, features), 1)
        pooled_output_with_src = pooled_output
        outp_with_f = F.relu(self.dropout(self.hidden_layer(pooled_output_with_src)))
        logits = self.last_layer(outp_with_f)

        return logits
