import torch
from matplotlib.pyplot import xlabel, ylabel, title
from torch import nn, autograd, distributions, exp, log
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from AIAV import AIAV
from AIAVDataset import SysData
from alive_progress import alive_bar

pd.set_option('future.no_silent_downcasting', True) # pandas is getting rid of fillna in the future
totalSyscalls = 456
emb_dim = 150 # HyperParameter (20 on AIAV.py)
seq_len = 40 # HyperParameter (100 on LAIAV)

def train(rnn, iterations=10000, load=DataLoader(SysData('slideshow_syscalls', 30, totalSyscalls))):
    rnn.to("cuda:0")
    rnn.train()
    i = 0
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.05)  # lr is a HyperParameter, but not that important as long as nothing "explodes"
    lossfunc = nn.MSELoss(reduction='sum')  # Sum differences
    ret = []
    with alive_bar(iterations, force_tty=True) as bar:
        for sequence, new in iter(load):
            if new:
                rnn.clear()
            sequence = sequence.squeeze()  # Undoes the batch layer
            sequence = sequence.to("cuda:0")
            output = rnn.forward(sequence)
            error = lossfunc(output, sequence)
            optimizer.zero_grad()  # If not included, as this loops previous grads will leak into next update
            error.backward()  # Calculate the gradients
            optimizer.step()  # Update the weights
            ret.append(error.item())
            i += 1
            if i > iterations:
                break
            bar()
    return ret

def test(rnn, iterations=100, load=DataLoader(SysData('slideshow_syscalls_test', 30, totalSyscalls))):
    lossfunc = nn.MSELoss(reduction='sum')  # Sum differences
    i = 0
    rnn.eval()
    ret = []
    with alive_bar(iterations, force_tty=True) as bar:
        with torch.no_grad():
            for inp, new in iter(load):
                if new:
                    rnn.clear()
                if inp is None:
                    break
                inp = inp.to("cuda:0")
                out = rnn.forward(inp)
                inp = inp.squeeze()  # Undoes the batch layer
                loss = lossfunc(out, inp)
                if loss > 32.367:
                    for ea in inp:
                        for j in range(len(ea)):
                            if(ea[j]):
                                print(j)
                    #print(inp)
                ret.append(loss.item())
                i += 1
                bar()
                if i > iterations:
                    break
    print(np.max(ret))
    return ret

if __name__=="__main__":
    #rnn.load_state_dict(torch.load('wulverMODEL.pt'))
    bests = 0
    beste = 0
    best = 1000
    rnn = AIAV(seq_len, emb_dim)
    rnn.to('cuda:0')
    trainls = train(rnn, iterations=10000, load=DataLoader(SysData('slideshow_syscalls', seq_len, totalSyscalls)))
    '''for s in range(10, 100, 5):
        seq_len = s
        for e in range(int(s/2), s, 2):
            emb_dim = e
            rnn = AIAV(seq_len, emb_dim)
            rnn.to('cuda:0')
            trainls = train(rnn, iterations=10000, load=DataLoader(SysData('slideshow_syscalls', seq_len, totalSyscalls)))
            if np.mean(trainls) < best:
                best = np.mean(trainls)
                rnn.save()
                bests = seq_len
                beste = e'''
    rnn.save()
    #rnn.load_state_dict(torch.load('wulverOneOut.pt'))
    testls = test(rnn, iterations=50, load=DataLoader(SysData('slideshow_syscalls', seq_len, totalSyscalls)))
    badls = test(rnn, iterations=50, load=DataLoader(SysData('badMacro_syscalls', seq_len, totalSyscalls)))
    open("results.txt", "w").write(f"testMean is {np.mean(testls)}, bad mean is {np.mean(badls)}, also s is {bests}, e is {beste}")
    #print(test_loss, train_loss)
    plt.plot(badls, label="Virus")
    plt.plot(testls, label="Not Virus")
    xlabel("Sequence Number")
    ylabel("Model Loss")
    title("Loss vs. Sequence for Virus and Not Virus files")
    plt.legend()
    plt.show()
