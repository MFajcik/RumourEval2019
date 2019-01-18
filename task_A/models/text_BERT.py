import logging

from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from torch import nn


def dump_batch_contents(batch):
    logging.debug("#" * 30)
    logging.debug("Dumping batch contents")
    for i in range(batch.text.shape[0]):
        logging.debug(f"L:{len(batch.text[i])} T: {batch.raw_text[i]}")


class BertModelForStanceClassification(PreTrainedBertModel):
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
        super(BertModelForStanceClassification, self).__init__(config)
        self.bert = BertModel(config)
        # hack token type embeddings
        # self.bert.embeddings.token_type_embeddings = torch.nn.Embedding(3,768)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.bertout_layer = nn.Linear(config.hidden_size, config.hidden_size)
        # self.hidden_layer = nn.Linear(config.hidden_size, config.hidden_size+20)
        # self.last_layer = nn.Linear(config.hidden_size+50, classes)
        self.last_layer = nn.Linear(config.hidden_size, classes)
        self.apply(self.init_bert_weights)

    def forward(self, batch):
        _, pooled_output = self.bert(batch.text, batch.type_mask, None, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        # source_feature =  batch.issource.float().unsqueeze(-1)
        # sentiment_features =  [batch.sentiment_pos.float().unsqueeze(-1),
        #                        batch.sentiment_neu.float().unsqueeze(-1),
        #                        batch.sentiment_neg.float().unsqueeze(-1)]

        # fs = [source_feature] + sentiment_features
        # features = torch.cat(
        #     fs, dim=-1)

        # pooled_output_with_src = torch.cat((pooled_output,features), -1)
        # transformed_outp = F.relu(self.dropout(self.bertout_layer(pooled_output)))
        #
        # inp_with_f = torch.cat((transformed_outp,batch.issource.float().unsqueeze(-1)),1)
        # outp_with_f = F.relu(self.dropout(self.hidden_layer(pooled_output)))
        # logits =  self.last_layer(outp_with_f)
        logits = self.last_layer(pooled_output)

        return logits

