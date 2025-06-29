import os, json
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from nilmtk.disaggregate import Disaggregator

class DAEModel(nn.Module):
    """
    Convolutional autoencoder for appliance load disaggregation.
    """
    def __init__(self, seq_len):
        super().__init__()
        # PyTorch 1.10+ supports padding="same"
        self.conv1   = nn.Conv1d(1,  8, kernel_size=4, padding="same")
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(seq_len*8, seq_len*8)
        self.fc2     = nn.Linear(seq_len*8, 128)
        self.fc3     = nn.Linear(128, seq_len*8)
        # unflatten back to (batch, channels=8, seq_len)
        self.seq_len = seq_len
        self.conv2   = nn.Conv1d(8, 1, kernel_size=4, padding="same")

    def forward(self, x):
        # x: [batch, seq_len, 1]
        x = x.permute(0,2,1)       # -> [batch, 1, seq_len]
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)        # -> [batch, 8*seq_len]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))# -> [batch, 8*seq_len]
        x = x.view(-1, 8, self.seq_len)  # -> [batch, 8, seq_len]
        x = self.conv2(x)          # -> [batch, 1, seq_len]
        x = x.permute(0,2,1)       # -> [batch, seq_len, 1]
        return x

class DAE(Disaggregator):
    def __init__(self, params):
        super().__init__()
        self.MODEL_NAME        = "DAE"
        self.file_prefix       = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.sequence_length   = params.get('sequence_length', 99)
        self.n_epochs          = params.get('n_epochs', 10)
        self.batch_size        = params.get('batch_size', 512)
        self.mains_mean        = params.get('mains_mean', 1000)
        self.mains_std         = params.get('mains_std', 600)
        self.appliance_params  = params.get('appliance_params', {})
        self.save_model_path   = params.get('save-model-path', None)
        self.load_model_path   = params.get('pretrained-model-path', None)
        self.models            = OrderedDict()
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.load_model_path:
            self.load_model()

    def return_network(self):
        return DAEModel(self.sequence_length).to(self.device)

    def set_appliance_params(self, train_appliances):
        """
        Set the mean and std for each appliance based on the training data.
        """
        for name, lst in train_appliances:
            arr = pd.concat(lst, axis=0).values.flatten()
            m, s = arr.mean(), arr.std()
            if s < 1: s = 100  # avoid zero std
            self.appliance_params[name] = {'mean': m, 'std': s}

    def normalize_input(self, data, n, mean, std, overlap):
        flat = data.flatten()
        pad  = (n - flat.size % n) % n
        flat = np.concatenate([flat, np.zeros(pad)])
        if overlap:
            # sliding windows
            w = np.array([flat[i:i+n] for i in range(len(flat)-n+1)])
        else:
            # non-overlapping windows
            w = flat.reshape(-1, n)
        return ((w - mean)/std).reshape(-1, n, 1)  # normalize and reshape for model

    def denormalize_output(self, data, mean, std):
        return mean + data*std

    def call_preprocessing(self, mains_lst, subs, method):
        """
        Preprocess the mains and appliances data for training or testing.
        """
        if method == 'train':
            pm, apps = [], []
            for mains in mains_lst:
                x = self.normalize_input(
                    mains.values,
                    self.sequence_length,
                    self.mains_mean,
                    self.mains_std,
                    True
                )
                pm.append(pd.DataFrame(x.reshape(-1, self.sequence_length)))
            for name, lst in subs:
                m,s = self.appliance_params[name]['mean'], self.appliance_params[name]['std']
                dfs = []
                for df in lst:
                    y = self.normalize_input(df.values, self.sequence_length, m, s, True)
                    dfs.append(pd.DataFrame(y.reshape(-1, self.sequence_length)))
                apps.append((name, dfs))
            return pm, apps

        # test mode
        pm = []
        for mains in mains_lst:
            x = self.normalize_input(
                mains.values,
                self.sequence_length,
                self.mains_mean,
                self.mains_std,
                False
            )
            pm.append(pd.DataFrame(x.reshape(-1, self.sequence_length)))
        return pm

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **_):
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train'
            )
        mains_arr = pd.concat(train_main, axis=0).values.reshape(-1, self.sequence_length, 1)

        apps = []
        for name, dfs in train_appliances:
            arr = pd.concat(dfs,axis=0).values.reshape(-1, self.sequence_length, 1)
            apps.append((name, arr))

        for name, arr in apps:
            if name not in self.models:
                self.models[name] = self.return_network()
            model = self.models[name]

            X = torch.tensor(mains_arr, dtype=torch.float32)  # mains input
            Y = torch.tensor(arr, dtype=torch.float32)  # appliance output
            split = int(len(X)*0.85)
            tr_ds = TensorDataset(X[:split], Y[:split])  # train set
            va_ds = TensorDataset(X[split:], Y[split:])  # validation set
            tr = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)  # train loader
            va = DataLoader(va_ds, batch_size=self.batch_size)  # validation loader

            opt     = optim.Adam(model.parameters())
            loss_fn = nn.MSELoss()
            best    = float('inf')
            ckpt    = f"{self.file_prefix}-{name.replace(' ','_')}-epoch{current_epoch}.pt"

            for _ in tqdm(range(self.n_epochs), desc=name):
                model.train()
                for xb, yb in tr:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    out = model(xb)
                    loss_fn(out, yb).backward()
                    opt.step()

                model.eval()
                vl = []
                with torch.no_grad():
                    for xb, yb in va:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        vl.append(loss_fn(model(xb), yb).item())
                val_loss = sum(vl)/len(vl)
                if val_loss < best:
                    best = val_loss
                    torch.save(model.state_dict(), ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))

        if self.save_model_path:
            self.save_model()

    def save_model(self):
        os.makedirs(self.save_model_path, exist_ok=True)
        params = {
            'sequence_length': self.sequence_length,
            'mains_mean':      self.mains_mean,
            'mains_std':       self.mains_std,
            'appliance_params':self.appliance_params
        }
        with open(os.path.join(self.save_model_path,'model.json'),'w') as f:
            json.dump(params, f)
        for name, m in self.models.items():
            torch.save(m.state_dict(),
                       os.path.join(self.save_model_path, f"{name}.pt"))

    def load_model(self):
        with open(os.path.join(self.load_model_path,'model.json')) as f:
            p = json.load(f)
        self.sequence_length = p['sequence_length']
        self.mains_mean      = p['mains_mean']
        self.mains_std       = p['mains_std']
        self.appliance_params= p['appliance_params']
        for name in self.appliance_params:
            m = self.return_network()
            m.load_state_dict(torch.load(
                os.path.join(self.load_model_path, f"{name}.pt"),
                map_location=self.device
            ))
            self.models[name] = m

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, None, 'test'
            )
        results = []
        for tm in test_main_list:
            arr = tm.values.reshape(-1, self.sequence_length, 1)
            ds  = DataLoader(
                TensorDataset(torch.tensor(arr, dtype=torch.float32)),
                batch_size=self.batch_size
            )
            outd = {}
            for name, m in self.models.items():
                preds = []
                m.eval()
                with torch.no_grad():
                    for (xb,) in ds:
                        xb = xb.to(self.device)
                        p  = m(xb).cpu().numpy()
                        preds.append(p)
                p_all = np.concatenate(preds).reshape(-1, self.sequence_length)
                mean,std = self.appliance_params[name].values()
                p_den = self.denormalize_output(p_all, mean, std).flatten()
                p_den = np.clip(p_den, 0, None)
                outd[name] = pd.Series(p_den)
            results.append(pd.DataFrame(outd, dtype='float32'))
        return results