import copy
import math

import torch


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

    def get_output_dim(self):
        raise NotImplementedError("Objects need to implement this method!")


class RNN(Encoder):
    def __init__(self, config):
        super(RNN, self).__init__(config)
        self.rnn = None

    def init_hidden(self, directions, initfun=torch.zeros, requires_grad=True):
        """
        Init RNN hidden state
        :param directions: number of directions
        :param initfun: function to initialize hidden state from,
                default: torch.randn, which provides samples from normal gaussian distribution (0 mean, 1 variance)
        :param requires_grad: if the hidden states should be learnable, default = True

        Initializes variable self.hidden
        """
        self.hidden_params = torch.nn.Parameter(
            initfun(self.layers * directions, 1, self.hidden_size, requires_grad=requires_grad)
        )
        self.cell_params = torch.nn.Parameter(
            initfun(self.layers * directions, 1, self.hidden_size, requires_grad=requires_grad))

    def forward(self, inp):
        """
        :param inp: Shape BATCH_SIZE x LEN x H_DIM
        """
        assert self.rnn
        bsz = inp.shape[0]
        hidden_params = self.hidden_params.repeat(1, bsz, 1)
        cell_params = self.cell_params.repeat(1, bsz, 1)
        outp = self.rnn(inp, (hidden_params, cell_params))[0]
        return outp

    def get_output_dim(self):
        return self.output_dim


class LSTM(RNN):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config['RNN_nhidden']
        self.layers = config['RNN_layers']
        self.rnn = torch.nn.LSTM(
            config["RNN_input_dim"],
            self.hidden_size, self.layers,
            dropout=config['dropout_rate'],
            batch_first=True,
            bidirectional=False)
        self.init_hidden(directions=1)
        self.output_dim = config['RNN_nhidden']


class BiLSTM(RNN):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config['RNN_nhidden']
        self.layers = config['RNN_layers']
        self.rnn = torch.nn.LSTM(
            config["RNN_input_dim"],
            self.hidden_size, self.layers,
            dropout=float(config['dropout_rate']),
            batch_first=True,
            bidirectional=True)
        self.init_hidden(directions=2)
        self.output_dim = config['RNN_nhidden'] * 2


class SelfAttentiveEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.rnn = BiLSTM(config)
        self.outputdim = int(config['RNN_nhidden']) * 2 * int(config['ATTENTION_hops'])
        self.nhops = int(config['ATTENTION_hops'])
        self.drop = torch.nn.Dropout(config['dropout_rate'])

        # The bias on these layers should be turned off according to paper!
        self.ws1 = torch.nn.Linear(int(config['RNN_nhidden']) * 2,
                                   int(config['ATTENTION_nhidden']),
                                   bias=False)

        self.ws2 = torch.nn.Linear(int(config['ATTENTION_nhidden']),
                                   self.nhops,
                                   bias=False)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def get_output_dim(self):
        return self.outputdim

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, emb, vocab):
        # outp has shape [len,bsz, nhid*2]
        outp = self.rnn.forward(emb).contiguous()
        batch_size, inp_len, h_size2 = outp.size()  # [bsz, len, nhid*2]
        # flatten dimension 1 and 2
        compressed_embeddings = outp.view(-1, h_size2)  # [bsz*len, nhid*2]

        # Calculate attention
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, nattention]
        alphas = self.ws2(hbar).view(batch_size, inp_len, -1)  # [bsz, len, hop]

        # Transpose input and reshape it
        transposed_inp = inp.view(batch_size, 1, inp_len)  # [bsz, 1, len]
        concatenated_inp = [transposed_inp for _ in range(self.nhops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        # Mask attention on padded sequence to zero
        alphas = torch.transpose(alphas, 1, 2).contiguous()
        padded_attention = -1e8 * (concatenated_inp == vocab.stoi['<pad>']).float()
        alphas += padded_attention

        talhpas = alphas.view(-1, inp_len)  # [bsz*hop,inp_len]
        # Softmax over 1st dimension (with len inp_len)
        alphas = self.softmax(talhpas)
        alphas = alphas.view(batch_size, self.nhops, inp_len)  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas


### Transformer transducer

def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TransformerEncoder(torch.nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # TODO: inputs to Sublayer are ALWAYS normalized, initial
        # Embeddings are normalized too!
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(torch.nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 2 because there are 2 sublayers in each block
        # 1st sublayers does multi-head attention
        # 2nd sublayer does position-wise feed forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, config, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        d_model = int(config['d_model'])
        heads = int(config['heads'])
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads  # dimensionality over one head
        self.heads = heads
        # 4 - for query, key, value transformation + output transformation
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, input, mask=None):
        return self.attention(input, input, input, mask)

    def attention(self, query, key, value, mask=None):
        "Implements Figure 2"
        # Query has shape BATCH x LEN x D_MODEL
        # Mask has shape BATCH x S x LEN
        # where S is:  1 for src sequence data
        #            LEN for tgt sequence data

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # mask is now BATCH x 1 x S x LEN
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        # If query,key,value are of size bsz x length x d_model
        # this code transforms query, key and value with d_model x d_model matrices
        # and splits each into bsz x h (number of splits) x length x d_k

        # Rewritten into more clear representation
        query = self.linears[0](query).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.scaled_dot_product_attention(query, key, value, mask=mask,
                                                         dropout=self.dropout)
        # x has shape bsz x length x d_model
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.heads * self.d_k)
        return self.linears[-1](x)

    def scaled_dot_product_attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        # compute similarity of query-to-key vectors via dot product
        # normalize it via length of dimension
        #
        # From the paper:
        # The two most commonly used attention functions are
        #   additive attention
        #   dot-product (multiplicative) attention.
        #
        # Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√d_k.
        # Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.
        # While the two are similar in theoretical complexity, dot-product attention is much faster and more
        # space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
        #
        # While for small values of d_k the two mechanisms perform similarly, additive attention outperforms
        # dot product attention without scaling for larger values of d_k. We suspect that for large values of
        # d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has
        # extremely small gradients
        #
        # To illustrate why the dot products get large, assume that the components of
        # q and k are independent random variables with mean 0 and variance 1. Then their dot product, q⋅k= i from {1 ... d_k} ∑qiki,
        # has mean 0 and variance d_k.
        # To counteract this effect, we scale the dot products by 1/√d_k.
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        # scores has shape
        # BATCH x HEADS x LEN_QUERY x LEN_KEY
        if mask is not None:
            # masked fill is broadcastable
            # dimensions 1 and 2 are broadcasted
            # mask through the dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        # Mask has shape BATCH x 1 x S x LEN
        # where S is:  1 for src sequence data
        #            LEN for tgt sequence data

        # NOTICE: Dropout on attention
        if dropout is not None:
            p_attn = dropout(p_attn)

        # The result is
        # KEY aware query representation
        # It will have length BATCH x HEADS x Query_LEN x d_k
        # where there is Query_LEN d_k vectors, each mixed from learned
        # weighted average of value vectors
        return torch.matmul(p_attn, value), p_attn

### End of transformer transducer
