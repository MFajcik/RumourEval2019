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
