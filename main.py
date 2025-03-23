import torch
from torch import nn, distributions, exp, log
import numpy as np
import pandas as pd
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import os

pd.set_option('future.no_silent_downcasting', True) # pandas is getting rid of fillna in the future
totalSyscalls = 457
emb_dim = 100 # HyperParameter
seq_len = 2000 # HyperParameter

class AIAV(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, seq_len, emb_dim):
            super().__init__()
            self.inp = totalSyscalls + 1
            self.seq_len = seq_len
            self.emb_dim = emb_dim
            self.model = nn.LSTM(
                input_size=self.inp,
                hidden_size=self.emb_dim,
                num_layers=1,
                batch_first=True # Tensors will be (batch, seq, features)
            )
        def encode(self, x):
            x = x.reshape((1, self.seq_len, self.inp))
            final, (hidden, cells) = self.model(x)  # Outputs Final, (hidden, cells)
            # We want the hidden layer because we will be reconstructing the model from it
            # Hidden is a 1xemb_dim torch tensor. We will be using this to reconstruct the input
            return hidden.squeeze()
    class Decoder(nn.Module):
        def __init__(self, seq_len, emb_dim):
            super().__init__()
            self.inp = emb_dim # Takes the hidden layer of decoder as input
            self.seq_len = seq_len
            self.emb_dim = totalSyscalls + 1
            self.model = nn.LSTM(
                input_size=self.inp,
                hidden_size=self.emb_dim,
                num_layers=1  # HyperParameter
            )
        def decode(self, x):
            x = x.repeat(self.seq_len, 1) # For every sequence
            x, (_, _) = self.model(x) # x now is the reconstructed representations of the sequences
            return x
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.encoder = self.Encoder(seq_len, emb_dim)
        self.decoder = self.Decoder(seq_len, emb_dim)
    def forward(self, x):
        hidden = self.encoder.encode(x) # Outputs Final, (hidden, cells)
        out = self.decoder.decode(hidden)
        return out

class SysData(Dataset):
    def updateData(self):
        try: # Overwrite old data (don't combine different syscall lines)
            self.cidx += 1
            if self.fidx == 1: # Just Created
                self.read = pd.read_csv(f'{self.dirname}/zpoline_syscalls_{self.fidx}.csv', chunksize=seq_len)
                self.fidx += 1
                self.cidx = 0
            data = next(self.read)
            if len(data) < seq_len:
                print('NEXT FILE')
                self.read = pd.read_csv(f'{self.dirname}/zpoline_syscalls_{self.fidx}.csv', chunksize=seq_len)
                data = next(self.read)
                self.fidx += 1
                self.cidx = 0
        except:
            self.data = None
            return
        data = data.drop(columns=['PID']) # FOR NOW
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
        data['TIMESTAMP'] = data['TIMESTAMP'].diff().fillna(timedelta(0)).apply(lambda x: x / timedelta(microseconds=1))
        syscalls = pd.DataFrame(columns=range(458), index=range(len(data)))
        syscalls = syscalls.fillna(0)
        for i in range(len(data)):
            syscalls.iloc[i, [data['SYSCALL_NUMBER'][i + seq_len*self.cidx]]] = 1
            syscalls.iloc[i, -1] = data['TIMESTAMP'][i + seq_len*self.cidx]
        self.data = syscalls
    def getLen(self):
        self.len = 0
        for i in range(len(os.listdir(self.dirname))):
            self.len += sum(1 for _ in open(f'{self.dirname}/zpoline_syscalls_{i+1}.csv', 'rb'))
    def __init__(self, dirname):
        self.fidx = 1 # File Index
        self.cidx = 0 # chunk index
        self.dirname = dirname
        self.read = None
        self.getLen()
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        self.updateData()
        if self.data is not None:
            itemAndLabel =  torch.tensor(self.data.iloc[0:seq_len, :].values.astype(np.float32)).unsqueeze(0)
            return itemAndLabel # Our item IS our label
        else:
            return None

if __name__=="__main__":
    rnn = AIAV(seq_len, emb_dim)
    #data = getData('zpoline_syscalls.csv')
    data = SysData('slideshow_syscalls')
    test = SysData('slideshow_syscalls')
    tload = DataLoader(test)
    load = DataLoader(data)
    # Training Doers
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3) # lr is a HyperParameter, but not that important as long as nothing "explodes"
    lossFunc = nn.L1Loss(reduction='sum') # Sum differences

    ls = []
    testls = []
    rnn.train()
    for inp in iter(load):
        out = rnn.forward(inp)
        inp = inp.squeeze() # Undoes the batch layer
        loss = lossFunc(out, inp)
        optimizer.zero_grad() # If not included, as this loops previous grads will leak into next update
        loss.backward() # Calculate the gradients
        optimizer.step() # Update the weights
        ls.append(loss.item())
    rnn.eval()
    with torch.no_grad():
        for inp in iter(tload):
            out = rnn.forward(inp)
            inp = inp.squeeze()  # Undoes the batch layer
            loss = lossFunc(out, inp)
            testls.append(loss.item())
    train_loss = np.mean(ls)
    test_loss = np.mean(testls)
    print(train_loss, test_loss)
#TODO: Train!
#TODO: Write the train/test split

