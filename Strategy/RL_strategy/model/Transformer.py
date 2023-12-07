import torch
import math
torch.manual_seed(1234)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if len(x.shape) == 3:
            position_encoding = self.pe[:, :seq_len, :].expand_as(x)
            x = x + position_encoding
        elif len(x.shape) == 2:
            x = x + self.pe[0, :x.size(0), :]
        return x

class Transformer(torch.nn.Module):
    def __init__(self, input_dim,output_dim,d_model=128,nhead=8,num_encoder_layers=8):
        # torch.manual_seed(2345)
        super(Transformer, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = torch.nn.Transformer(d_model, nhead, num_encoder_layers)
        self.fc = torch.nn.Linear(d_model, output_dim)

    def forward(self, sequence):
        embedded_sequence = self.embedding(sequence)
        embedded_sequence = self.pos_encoder(embedded_sequence)

        if len(embedded_sequence.shape) == 3:
        # [sequence_length, batch_size, d_model]
            transformer_input = embedded_sequence.permute(1, 0, 2)
            transformer_output = self.transformer(transformer_input, transformer_input)
        # [batch_size, sequence_length, d_model]
            transformer_output = transformer_output.permute(1, 0, 2)
            pooled_output = transformer_output.mean(dim=1)
        # pooled_output 形状：[batch_size, d_model]

        elif len(embedded_sequence.shape) == 2:
            transformer_output = self.transformer(embedded_sequence, embedded_sequence)
            pooled_output = transformer_output.mean(dim=0)
        
        output =  self.fc(pooled_output)
        return output