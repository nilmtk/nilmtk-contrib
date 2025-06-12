import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilmtk.disaggregate import Disaggregator

class GRUNet(nn.Module):
    def __init__(self, sequence_length):
        super(GRUNet, self).__init__()
        self.conv1    = nn.Conv1d(1, 16, kernel_size=4, padding=2)
        self.gru1     = nn.GRU(16, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2     = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1      = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(128, 1)

    def forward(self, x):
        # x: [batch, 1, seq_len]
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)        # [batch, seq_len, 16]
        x, _   = self.gru1(x)         # [batch, seq_len, 128]
        x      = self.dropout1(x)
        _, h_n = self.gru2(x)         # h_n: [2, batch, 128]
        h      = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, 256]
        h      = self.dropout2(h)
        h      = self.fc1(h)          # [batch, 128]
        h      = torch.relu(h)
        h      = self.dropout3(h)
        out    = self.fc2(h)          # [batch, 1]
        return out

class WindowGRU(Disaggregator):
    def __init__(self, params):
        super().__init__()
        self.MODEL_NAME      = "WindowGRU"
        self.file_prefix     = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path', None)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs        = params.get('n_epochs', 10)
        self.batch_size      = params.get('batch_size', 512)
        self.max_val         = 800
        self.models          = OrderedDict()
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def return_network(self):
        return GRUNet(self.sequence_length).to(self.device)

    def partial_fit(self, train_main, train_appliances,
                    do_preprocessing=True, current_epoch=0, **kwargs):
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train'
            )

        mains_arr = pd.concat(train_main, axis=0).values \
                    .reshape(-1, self.sequence_length)  # [N, seq_len]
        new_apps = []
        for app_name, df_list in train_appliances:
            concatenated = pd.concat(df_list, axis=0)
            arr = concatenated.values.reshape(-1, 1)      # [N, 1]
            new_apps.append((app_name, arr))

        for app_name, arr in new_apps:
            if app_name not in self.models:
                self.models[app_name] = self.return_network()
            model = self.models[app_name]

            x_cpu = torch.tensor(mains_arr, dtype=torch.float32)
            y_cpu = torch.tensor(arr, dtype=torch.float32)
            split = int(len(x_cpu) * 0.85)

            train_ds = TensorDataset(x_cpu[:split], y_cpu[:split])
            val_ds   = TensorDataset(x_cpu[split:], y_cpu[split:])
            train_loader = DataLoader(train_ds,
                                      batch_size=self.batch_size,
                                      shuffle=True)
            val_loader   = DataLoader(val_ds,
                                      batch_size=self.batch_size)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            best_val  = float('inf')
            ckpt_path = f"{self.file_prefix}-{app_name.replace(' ','_')}-epoch{current_epoch}.pt"

            for epoch in tqdm(range(self.n_epochs),
                              desc=f"Train {app_name}"):
                model.train()
                for xb_cpu, yb_cpu in train_loader:
                    xb = xb_cpu.unsqueeze(1).to(self.device)  # [B,1,seq]
                    yb = yb_cpu.to(self.device)               # [B,1]
                    optimizer.zero_grad()
                    out = model(xb)                           # [B,1]
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb_cpu, yb_cpu in val_loader:
                        xb = xb_cpu.unsqueeze(1).to(self.device)
                        yb = yb_cpu.to(self.device)
                        out = model(xb)
                        val_losses.append(criterion(out, yb).item())
                val_loss = sum(val_losses) / len(val_losses)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), ckpt_path)
            model.load_state_dict(torch.load(ckpt_path,
                                             map_location=self.device))
            torch.cuda.empty_cache()

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, None, 'test'
            )

        results = []
        for mains in test_main_list:
            arr = mains.values.reshape(-1, self.sequence_length)
            x_cpu = torch.tensor(arr, dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(x_cpu),
                                     batch_size=self.batch_size)
            out_dict = {}
            for app_name, m in self.models.items():
                preds = []
                m.eval()
                with torch.no_grad():
                    for (xb_cpu,) in test_loader:
                        xb = xb_cpu.unsqueeze(1).to(self.device)
                        p  = m(xb).view(-1).cpu().numpy()
                        preds.append(p)
                all_pred = np.concatenate(preds)
                all_pred = np.clip(all_pred, 0, None) * self.max_val
                out_dict[app_name] = pd.Series(all_pred)
                torch.cuda.empty_cache()
            results.append(pd.DataFrame(out_dict, dtype='float32'))
        return results

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == 'train':
            pm, apps = [], []
            for mains in mains_lst:
                pad = [0] * (self.sequence_length - 1)
                tmp = pd.concat([mains,
                                 pd.DataFrame({mains.columns[0]: pad})])
                pm.append(pd.DataFrame(self.preprocess_train_mains(tmp)))
            for name, lst in submeters_lst:
                dfs = [pd.DataFrame(self.preprocess_train_appliances(df))
                       for df in lst]
                apps.append((name, dfs))
            return pm, apps

        if method == 'test':
            pm = []
            for mains in mains_lst:
                pad = [0] * (self.sequence_length - 1)
                tmp = pd.concat([mains,
                                 pd.DataFrame({mains.columns[0]: pad})])
                pm.append(pd.DataFrame(self.preprocess_test_mains(tmp)))
            return pm

    def preprocess_train_mains(self, mains):
        arr = (mains / self.max_val).values
        idx = (np.arange(self.sequence_length)[None, :]
               + np.arange(len(arr) - self.sequence_length + 1)[:, None])
        return arr[idx].reshape(-1, self.sequence_length)

    def preprocess_train_appliances(self, app):
        return (app / self.max_val).values.reshape(-1, 1)

    def preprocess_test_mains(self, mains):
        arr = (mains / self.max_val).values
        idx = (np.arange(self.sequence_length)[None, :]
               + np.arange(len(arr) - self.sequence_length + 1)[:, None])
        return arr[idx].reshape(-1, self.sequence_length)

    def _normalize(self, chunk, m):
        return chunk / m

    def _denormalize(self, chunk, m):
        return chunk * m


          