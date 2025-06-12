import os, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from nilmtk.disaggregate import Disaggregator

class Seq2SeqModel(nn.Module):
    def __init__(self, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(1, 30, 10, stride=2)
        self.conv2 = nn.Conv1d(30,30, 8,  stride=2)
        self.conv3 = nn.Conv1d(30,40, 6,  stride=1)
        self.conv4 = nn.Conv1d(40,50, 5,  stride=1)
        self.dropout1 = nn.Dropout(.2)
        self.conv5 = nn.Conv1d(50,50, 5, stride=1)
        self.dropout2 = nn.Dropout(.2)

        # compute final time‐dimension after all conv/strides:
        L = seq_len
        L = (L - 10)//2 + 1
        L = (L - 8)//2 + 1
        L = L - 6 + 1
        L = L - 5 + 1
        L = L - 5 + 1
        flat_size = 50 * L

        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(flat_size, 1024)
        self.dropout3 = nn.Dropout(.2)
        self.fc2      = nn.Linear(1024, seq_len)

    def forward(self, x):
        # x: [B, seq_len, 1] → [B, 1, seq_len]
        x = x.permute(0,2,1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv5(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)           # [B, seq_len]
        return x

class Seq2Seq(Disaggregator):
    def __init__(self, params):
        super().__init__()

        self.MODEL_NAME = "Seq2Seq"
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        self.sequence_length     = params.get('sequence_length', 99)
        if self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd")
        self.n_epochs            = params.get('n_epochs', 10)
        self.batch_size          = params.get('batch_size', 512)
        self.mains_mean          = 1800
        self.mains_std           = 600
        self.appliance_params    = params.get('appliance_params', {})
        self.models              = OrderedDict()
        self.device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def return_network(self):
        return Seq2SeqModel(self.sequence_length).to(self.device)

    def set_appliance_params(self, train_appliances):
        for name, lst in train_appliances:
            arr = pd.concat(lst, axis=0).values.flatten()
            m, s = arr.mean(), arr.std()
            if s < 1: s = 100
            self.appliance_params[name] = {'mean':m, 'std':s}

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        n = self.sequence_length
        pad = n//2
        if method == 'train':
            pm, apps = [], []
            for mains in mains_lst:
                data = mains.values.flatten()
                data = np.pad(data,(pad,pad),'constant')
                windows = np.array([data[i:i+n] for i in range(len(data)-n+1)])
                normed = (windows - self.mains_mean)/self.mains_std
                pm.append(pd.DataFrame(normed))
            for name, lst in submeters_lst:
                if name not in self.appliance_params:
                    raise KeyError(f"No params for {name}")
                m,s = self.appliance_params[name]['mean'], self.appliance_params[name]['std']
                dfs = []
                for df in lst:
                    data = df.values.flatten()
                    data = np.pad(data,(pad,pad),'constant')
                    windows = np.array([data[i:i+n] for i in range(len(data)-n+1)])
                    normed = (windows - m)/s
                    dfs.append(pd.DataFrame(normed))
                apps.append((name, dfs))
            return pm, apps
        else:
            pm = []
            for mains in mains_lst:
                data = mains.values.flatten()
                windows = np.array([data[i:i+n] for i in range(len(data)-n+1)])
                normed = (windows - self.mains_mean)/self.mains_std
                pm.append(pd.DataFrame(normed))
            return pm

    def partial_fit(self, train_main, train_appliances,
                    do_preprocessing=True, current_epoch=0, **_):
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train'
            )

        mains_arr = pd.concat(train_main,axis=0).values \
                     .reshape(-1, self.sequence_length, 1)

        for name, dfs in train_appliances:
            arr = pd.concat(dfs,axis=0).values \
                    .reshape(-1, self.sequence_length)
            if name not in self.models:
                self.models[name] = self.return_network()
            model = self.models[name]

            X = torch.tensor(mains_arr, dtype=torch.float32)
            Y = torch.tensor(arr,       dtype=torch.float32)
            split = int(0.85*len(X))

            tr_ds = TensorDataset(X[:split], Y[:split])
            va_ds = TensorDataset(X[split:], Y[split:])
            tr = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
            va = DataLoader(va_ds, batch_size=self.batch_size)

            opt     = optim.Adam(model.parameters())
            loss_fn = nn.MSELoss()
            best    = float('inf')
            ckpt    = f"{self.file_prefix}-{name}-epoch{current_epoch}.pt"

            for epoch in tqdm(range(self.n_epochs), desc=f"Train {name}"):
                model.train()
                for xb, yb in tr:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    out = model(xb)                   # [B, seq_len]
                    loss_fn(out, yb).backward()
                    opt.step()

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in va:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        val_losses.append(loss_fn(model(xb), yb).item())
                val_loss = sum(val_losses)/len(val_losses)
                if val_loss < best:
                    best = val_loss
                    torch.save(model.state_dict(), ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model: self.models = model
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, None, 'test'
            )

        results = []
        n = self.sequence_length
        for tm in test_main_list:
            arr = tm.values.reshape(-1, n)
            ds  = DataLoader(TensorDataset(torch.tensor(arr, dtype=torch.float32)),
                             batch_size=self.batch_size)
            outd = {}
            for name, m in self.models.items():
                preds = []
                m.eval()
                with torch.no_grad():
                    for (xb_cpu,) in ds:
                        # Unsqueeze back to [B, seq_len, 1]
                        xb = xb_cpu.unsqueeze(-1).to(self.device)
                        p  = m(xb).cpu().numpy()    # [B, seq_len]
                        preds.append(p)
                P = np.concatenate(preds, axis=0)
                total = P.shape[0] + n - 1
                sum_arr    = np.zeros(total)
                counts_arr = np.zeros(total)
                for i in range(P.shape[0]):
                    sum_arr[i:i+n]    += P[i]
                    counts_arr[i:i+n]+= 1
                avg = sum_arr/counts_arr
                mpar = self.appliance_params[name]
                out  = mpar['mean'] + avg * mpar['std']
                outd[name] = pd.Series(np.clip(out, 0, None))
            results.append(pd.DataFrame(outd, dtype='float32'))
        return results
