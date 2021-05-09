from torch import nn


class MT_LSTM(nn.Module):

    def __init__(self, args):
        super(MT_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=args.input_size,
                               hidden_size=args.hidden_size,
                               num_layers=args.num_layers,
                               batch_first=True,
                               dropout=0.5
                               )
        self.decoder = nn.LSTM(input_size=args.input_size,
                               hidden_size=args.hidden_size,
                               num_layers=args.num_layers,
                               batch_first=True,
                               dropout=0.5
                               )
        self.fc = nn.Linear(args.hidden_size, args.embedding_size)

    def forward(self, encoder_inputs, encoder_hidden, encoder_cell, decoder_inputs):
        # h_t : [batch_size, num_layers * num_directions(=1), n_hidden]
        # c_t : [batch_size, num_layers * num_directions(=1), n_hidden]
        _, (h_t, c_t) = self.encoder(encoder_inputs, (encoder_hidden, encoder_cell))
        # outputs : [batch_size, dim_seq+1, num_directions(=1) * n_hidden(=128)]
        outputs, (_, _) = self.decoder(decoder_inputs, (h_t, c_t))
        # y: [batch_size, dim_seq+1, dim_embedding(=1)]
        y = self.fc(outputs)
        return y, c_t
