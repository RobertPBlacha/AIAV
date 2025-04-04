import torch
from matplotlib.pyplot import xlabel, ylabel, title
from torch import nn, autograd, distributions, exp, log
import numpy as np
import pandas as pd
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import os
import gc
import matplotlib.pyplot as plt
from alive_progress import alive_bar

pd.set_option('future.no_silent_downcasting', True) # pandas is getting rid of fillna in the future
totalSyscalls = 456
newFile = 0
emb_dim = 40 # HyperParameter (20 on AIAV)
seq_len = 30 # HyperParameter (100 on LAIAV)

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
            x, (hidden, _) = self.model(x)  # Outputs Final, (hidden, cells)
            # We want the hidden layer because we will be reconstructing the model from it
            out = hidden.squeeze(1)
            return out
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
            x = x.reshape((self.seq_len, self.inp))
            x, hidden = self.model(x) # x now is the reconstructed representations of the sequences
            return x
    def __init__(self, seq_len, emb_dim):
        super().__init__()
        self.encoder = self.Encoder(seq_len, emb_dim)
        self.decoder = self.Decoder(seq_len, emb_dim)
    def forward(self, x):
        hidden = self.encoder.encode(x) # Outputs Final, (hidden, cells)
        out = self.decoder.decode(hidden)
        return out
    def save(self):
        torch.save(self.state_dict(), './v2SLAIAVMODEL.pt')
    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)


class SysData(Dataset):
    def updateData(self):
        try: # Overwrite old data (don't combine different syscall lines)
            self.cidx += 1
            if self.fidx == 1: # Just Created
                self.read = pd.read_csv(f'{self.dirname}/zpoline_syscalls_{self.fidx}.csv', chunksize=seq_len)
                self.fidx += 1
                self.cidx = 0
                #print("First")
            data = next(self.read)
            if len(data) < seq_len:
                #print('NEXT FILE')
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
        syscalls = pd.DataFrame(columns=range(totalSyscalls + 1), index=range(len(data)))
        syscalls = syscalls.fillna(0)
        for i in range(len(data)):
            syscalls.iloc[i, [data['SYSCALL_NUMBER'][i + seq_len*self.cidx]]] = 1
            #syscalls.iloc[i, -1] = data['TIMESTAMP'][i + seq_len*self.cidx]
        self.data = syscalls
        gc.collect()
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
    print(torch.cuda.is_available())
    rnn = AIAV(seq_len, emb_dim)
    rnn.load_state_dict(torch.load('AIAV_v2_MODEL.pt'))
    rnn.to("cuda:0")
    data = SysData('slideshow_syscalls')
    test = SysData('slideshow_syscalls_test')
    tload = DataLoader(test)
    load = DataLoader(data)
    # Training Doers
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.05) # lr is a HyperParameter, but not that important as long as nothing "explodes"
    lossFunc = nn.MSELoss(reduction='sum') # Sum differences
    ls = []
    testls = []
    rnn.train()
    i = 0
    with alive_bar(10001, force_tty=True) as bar:
        for inp in iter(load):
            break
            inp = inp.squeeze()  # Undoes the batch layer
            inp = inp.to("cuda:0")
            out = rnn.forward(inp)
            loss = lossFunc(out, inp)
            optimizer.zero_grad() # If not included, as this loops previous grads will leak into next update
            loss.backward() # Calculate the gradients
            optimizer.step() # Update the weights
            ls.append(loss.item())
            i += 1
            bar()
            if i > 10000:
                break
    #rnn.save()
    rnn.eval()
    i= 0
    with alive_bar(51, force_tty=True) as bar:
        with torch.no_grad():
            for inp in iter(tload):
                if inp is None:
                    break
                inp = inp.to("cuda:0")
                out = rnn.forward(inp)
                inp = inp.squeeze()  # Undoes the batch layer
                loss = lossFunc(out, inp)
                testls.append(loss.item())
                i += 1
                bar()
                if i > 50:
                    break
    #train_loss = np.mean(ls)
    bload = SysData('badMacro_syscalls')
    badls = []
    i = 0
    with alive_bar(51, force_tty=True) as bar:
        with torch.no_grad():
            for inp in iter(bload):
                if inp is None:
                    break
                inp = inp.to("cuda:0")
                out = rnn.forward(inp)
                inp = inp.squeeze()  # Undoes the batch layer
                loss = lossFunc(out, inp)
                badls.append(loss.item())
                if loss < 25:
                    for j in range(len(inp)):
                        for k in range(len(inp[j])):
                            if inp[j][k] == 1:
                                print(k)
                                print(out[j][k])
                i += 1
                bar()
                if i > 50:
                    break
    #print(i, "bad iterations")
    test_loss = np.mean(testls)
    badLoss = np.mean(badls)
    print(test_loss, "<", badLoss)
    #print(test_loss, train_loss)
    plt.plot(badls, label="Virus")
    plt.plot(testls, label="Not Virus")
    xlabel("Sequence Number")
    ylabel("Model Loss")
    title("Loss vs. Sequence for Virus and Not Virus files")
    plt.legend()
    plt.show()
