import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import timedelta

totalSyscalls = 457
class AIAV(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.RNN(totalSyscalls+1, totalSyscalls)
        # RNN handles the tanh linearity, could consider switching to ReLu
        self.layer2 = nn.Linear(totalSyscalls, 1)
        self.act2 = nn.Sigmoid()
        self.hidden = torch.zeros(1, totalSyscalls) # Free to add more layers but probably not?

    def forward(self, x):
        x, self.hidden = self.layer1(x, self.hidden)
        #x = self.act1(x) TANH already RNN'd
        x = self.act2(self.layer2(x))
        return x



rnn = AIAV()  # consider changing nonlinearity to relu

data = pd.read_csv('zpoline_syscalls.csv')
data = data.drop(columns=['PID'])
data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
data['TIMESTAMP'] = data['TIMESTAMP'].diff().fillna(timedelta(0)).apply(lambda x: x / timedelta(microseconds=1))
syscalls = pd.DataFrame(columns=range(458), index=range(len(data)))
syscalls = syscalls.fillna(0)
for i in range(len(data)):
    syscalls.iloc[i, [data['SYSCALL_NUMBER'][i]]] = 1
    syscalls.iloc[i, -1] = data['TIMESTAMP'][i]
print(syscalls.head(10))
inp = torch.tensor(syscalls.head(10).values.astype(np.float32))
print(rnn.forward(inp))

#TODO: Train!

