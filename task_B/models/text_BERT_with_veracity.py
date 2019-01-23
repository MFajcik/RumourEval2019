import logging
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


class BertModelForVeracityClassification(PreTrainedBertModel):
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
        super(BertModelForVeracityClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.bertout_layer = nn.Linear(config.hidden_size, config.hidden_size)
        # self.hidden_layer = nn.Linear(config.hidden_size+1, config.hidden_size+50)
        # self.last_layer = nn.Linear(config.hidden_size+50, classes)
        self.last_layer_veracity = nn.Linear(config.hidden_size, 3)
        self.apply(self.init_bert_weights)
        self.max_batch_len = 0

    def forward(self, batch):
        _, pooled_output = self.bert(batch.text, batch.type_mask, None, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        # transformed_outp = F.relu(self.dropout(self.bertout_layer(pooled_output)))
        #
        # inp_with_f = torch.cat((transformed_outp,batch.issource.float().unsqueeze(-1)),1)
        # outp_with_f = F.relu(self.dropout(self.hidden_layer(inp_with_f)))
        # logits =  self.last_layer(outp_with_f)
        veracity_logits = self.last_layer_veracity(pooled_output)

        return veracity_logits
