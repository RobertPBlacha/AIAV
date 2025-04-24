from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import os
class SysData(Dataset):
    def updateData(self):
        try: # Overwrite old data (don't combine different syscall lines)
            self.cidx += 1
            self.new = 0
            if self.fidx == 1: # Just Created
                self.new = 1
                self.read = pd.read_csv(f'{self.dirname}/zpoline_syscalls_{self.fidx}.csv', chunksize=self.seq_len)
                self.fidx += 1
                self.cidx = 0
                #print("First")
            data = next(self.read)
            if len(data) < self.seq_len:
                #print('NEXT FILE')
                self.new = 1
                self.read = pd.read_csv(f'{self.dirname}/zpoline_syscalls_{self.fidx}.csv', chunksize=self.seq_len)
                data = next(self.read)
                self.fidx += 1
                self.cidx = 0
        except:
            self.data = None
            return
        data = data.drop(columns=['PID']) # FOR NOW
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
        data['TIMESTAMP'] = data['TIMESTAMP'].diff().fillna(timedelta(0)).apply(lambda x: x / timedelta(microseconds=1))
        syscalls = pd.DataFrame(columns=range(self.totalSyscalls + 1), index=range(len(data)))
        syscalls = syscalls.fillna(0)
        for i in range(len(data)):
            syscalls.iloc[i, [data['SYSCALL_NUMBER'][i + self.seq_len*self.cidx]]] = 1
            #syscalls.iloc[i, -1] = data['TIMESTAMP'][i + seq_len*self.cidx]
        self.data = syscalls
    def getLen(self):
        self.len = 0
        for i in range(len(os.listdir(self.dirname))):
            self.len += sum(1 for _ in open(f'{self.dirname}/zpoline_syscalls_{i+1}.csv', 'rb'))
    def __init__(self, dirname, seq_len, totalSyscalls):
        self.fidx = 1 # File Index
        self.cidx = 0 # chunk index
        self.new = 0
        self.dirname = dirname
        self.read = None
        self.seq_len = seq_len
        self.totalSyscalls = totalSyscalls
        self.getLen()
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        self.updateData()
        if self.data is not None:
            itemAndLabel =  torch.tensor(self.data.iloc[0:self.seq_len, :].values.astype(np.float32)).unsqueeze(0)
            return itemAndLabel, self.new # Our item IS our label
        else:
            return None