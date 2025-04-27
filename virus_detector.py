import torch
from matplotlib.pyplot import xlabel, ylabel, title
from torch import nn, autograd, distributions, exp, log
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import os
import gc
import matplotlib.pyplot as plt
import subprocess
import time
import tkinter as tk
from threading import Thread
from io import StringIO
from AIAV_CPU import AIAV, totalSyscalls

emb_dim = 100 # HyperParameter (20 on AIAV)
seq_len = 30 # HyperParameter (100 on LAIAV)

def monitor_live_csv(model_path='wulverMODELOne.pt', csv_path='/tmp/zpoline_syscalls_1.csv'):
    model = AIAV(seq_len, emb_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.to("cuda:0") NOT AVAILABLE ON KALI VM
    model.to("cpu")
    model.eval()
    threshold = 40  # Adjust based on empirical testing

    dataset = LiveSysData(csv_path, seq_len=seq_len)
    lossFunc = nn.MSELoss(reduction='sum')

    while True:
        window = dataset.get_next_window()
        if window is None:
            time.sleep(0.5)
            continue
        # window = window.to("cuda:0") NOT AVAILABLE ON KALI VM
        window = window.to("cpu")
        with torch.no_grad():
            model.clear()
            out = model.forward(window)
            loss = lossFunc(out, window.squeeze())
            print(f"Loss: {loss.item()}")
            if loss.item() > threshold:
                alert_popup(f"⚠️ Anomaly Detected!\nLoss: {loss.item():.5f}")
        time.sleep(1)  # You can tune this


def alert_popup(message="Anomaly detected!"):
    def show():
        root = tk.Tk()
        root.title("⚠️ Alert")
        label = tk.Label(root, text=message, padx=20, pady=20, font=("Helvetica", 14))
        label.pack()
        button = tk.Button(root, text="Dismiss", command=root.destroy, padx=10, pady=5)
        button.pack()
        root.mainloop()

    Thread(target=show).start()  # Run in new thread so it doesn’t block


class LiveSysData:
    def __init__(self, filepath, seq_len=40):
        self.filepath = filepath
        self.seq_len = seq_len
        self.last_idx = 0
        self.last_line = 0
        self.buffer = pd.DataFrame(columns=["SYSCALL_NUMBER", "PID", "TIMESTAMP"])

    def get_next_window(self):
        try:
            # Read ONLY the new lines
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
                new_lines = lines[self.last_line:]
                self.last_line = len(lines)

            new_lines = [line for line in new_lines if not line.strip().startswith('TIMESTAMP')]

            if not new_lines:
                return None
            
            # Parse new lines into dataframe.
            df_new = pd.read_csv(
                StringIO(''.join(new_lines)),
                header=None,
                names=["TIMESTAMP", "SYSCALL_NUMBER", "PID"]                                
            )
            self.buffer = pd.concat([self.buffer, df_new], ignore_index=True)

            # Drop PID and convert timestamp deltas
            self.buffer = self.buffer.drop(columns=['PID'])
            self.buffer['TIMESTAMP'] = pd.to_datetime(self.buffer['TIMESTAMP'])
            self.buffer['TIMESTAMP'] = self.buffer['TIMESTAMP'].diff().fillna(timedelta(0)).apply(
                lambda x : x / timedelta(microseconds=1)
            )

            # If not enough rows, return nothing
            if len(self.buffer) < self.seq_len:
                return None

            # Get most recent window
            window = self.buffer.iloc[:self.seq_len]
            self.buffer = self.buffer.iloc[1:]

            # Create one-hot encoded syscall vectors
            syscalls = pd.DataFrame(index=range(len(window)), columns=range(totalSyscalls + 1))
            syscalls = syscalls.fillna(0)
            for i in range(len(window)):
                syscall_num = window['SYSCALL_NUMBER'].iloc[i]
                if syscall_num < totalSyscalls + 1:
                    syscalls.iloc[i, syscall_num] = 1
                
            self.last_idx += 1  # Slide window by 1
            return torch.tensor(syscalls.values.astype(np.float32)).unsqueeze(0)  # Shape: (1, seq_len, features)
        except Exception as e:
            print(f"[ERROR] get_next_window: {e}")
            return None

if __name__ == "__main__":
    try:
        subprocess.Popen(["bash", "./setup.sh"])
        time.sleep(30)
        monitor_live_csv()
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped by user.")