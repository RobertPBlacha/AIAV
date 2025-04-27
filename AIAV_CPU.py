from torch import nn
totalSyscalls = 456
import torch

class AIAV(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, seq_len, emb_dim):
            super().__init__()
            self.inp = totalSyscalls + 1
            self.seq_len = seq_len
            self.emb_dim = emb_dim
            self.hidden = (torch.zeros(1, 1, self.emb_dim).to("cpu"), torch.zeros(1, 1, self.emb_dim).to("cpu"))
            self.model = nn.LSTM(
                input_size=self.inp,
                hidden_size=self.emb_dim,
                num_layers=1,
                batch_first=True # Tensors will be (batch, seq, features)
            )
        def encode(self, x):
            x = x.reshape((1, self.seq_len, self.inp))
            hidden, cells = self.hidden
            for i in x[0]: # for every sequence
                _, (hidden, cells) = self.model(i.unsqueeze(-2).unsqueeze(-2), (hidden, cells))  # Outputs Final, (hidden, cells)
            # We want the hidden layer because we will be reconstructing the model from it
            out = hidden.squeeze(1)
            self.hidden = (hidden.clone().detach(), cells.clone().detach())
            return out
        def clear(self):
            self.hidden = (torch.zeros(1, 1, self.emb_dim).to("cpu"), torch.zeros(1, 1, self.emb_dim).to("cpu"))
    class Decoder(nn.Module):
        def __init__(self, seq_len, emb_dim):
            super().__init__()
            self.inp = emb_dim # Takes the hidden layer of decoder as input
            self.seq_len = seq_len
            self.emb_dim = totalSyscalls + 1
            self.hidden = (torch.zeros(1, self.inp).to("cpu"), torch.zeros(1, self.emb_dim).to("cpu"))
            self.model = nn.LSTM(
                input_size=self.inp,
                hidden_size=self.emb_dim,
                proj_size=self.inp,
                num_layers=1  # HyperParameter
            )
        def clear(self):
            self.hidden = (torch.zeros(1, self.inp).to("cpu"), torch.zeros(1, self.emb_dim).to("cpu"))
        def decode(self, x):
            out = torch.zeros(0,457,device=x.device)
            hidden, cells = self.hidden
            for i in range(self.seq_len):
                x, (hidden, cells) = self.model(x, (hidden, cells))
                out = torch.cat([cells, out], 0)
            self.hidden = (hidden.clone().detach(), cells.clone().detach())
            return out
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.encoder = self.Encoder(seq_len, emb_dim)
        self.decoder = self.Decoder(seq_len, emb_dim)
        self.final = nn.Softmax(dim=1)
    def forward(self, x):
        hidden = self.encoder.encode(x) # Outputs Final, (hidden, cells)
        out = self.decoder.decode(hidden)
        return self.final(out)
    def clear(self):
        self.encoder.clear()
        self.decoder.clear()
    def save(self):
        torch.save(self.state_dict(), './wulverMODELOne.pt')
    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
