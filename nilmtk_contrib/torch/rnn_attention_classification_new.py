from __future__ import annotations
import copy, numpy as np, pandas as pd
from collections import OrderedDict
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm                                 

from nilmtk.disaggregate import Disaggregator
from nilmtk_contrib.torch.preprocessing import preprocess


class SequenceLengthError(Exception):
    pass


class ApplianceNotFoundError(Exception):
    pass


class IdentityBlock(nn.Module):
    def __init__(self, ch: int, k: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding="same")
        self.c2 = nn.Conv1d(ch, ch, k, padding="same")
        self.c3 = nn.Conv1d(ch, ch, k, padding="same")
        self.act = nn.ReLU()

    def forward(self, x):
        s = x
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.c3(x)
        return self.act(x + s)


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_mid: int, ch_out: int, k: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch_in,  ch_mid, k, padding="same")
        self.c2 = nn.Conv1d(ch_mid, ch_mid, k, padding="same")
        self.c3 = nn.Conv1d(ch_mid, ch_out, k, padding="same")
        self.proj = nn.Conv1d(ch_in, ch_out, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        s = self.proj(x)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.c3(x)
        return self.act(x + s)


class AttentionLayer(nn.Module):
    """Additive (Bahdanau) attention over the Bi-LSTM outputs."""
    def __init__(self, units: int):
        super().__init__()
        self.W = nn.Linear(units * 2, units)   # *2 : bidirectional
        self.V = nn.Linear(units, 1)

    def forward(self, enc_out):               # (B, T, 2H)
        score = self.V(torch.tanh(self.W(enc_out)))   # (B,T,1)
        weights = torch.softmax(score, dim=1)         # (B,T,1)
        ctx = torch.sum(weights * enc_out, dim=1)     # (B,2H)
        return ctx, weights.squeeze(-1)               # (B,2H), (B,T)


class _RNNAttNet(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

        self.cls_feat = nn.Sequential(
            nn.Conv1d(1, 30, 10), nn.ReLU(),
            nn.Conv1d(30, 30, 8), nn.ReLU(),
            nn.Conv1d(30, 40, 6), nn.ReLU(),
            nn.Conv1d(40, 50, 5), nn.ReLU(),
            nn.Conv1d(50, 50, 5), nn.ReLU(),
            nn.Conv1d(50, 50, 5), nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(1024), nn.ReLU()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(1024, seq_len),
            nn.Sigmoid()
        )

        self.conv_reg = nn.Conv1d(1, 16, 4, padding="same")
        self.bi1 = nn.LSTM(16, 128, batch_first=True, bidirectional=True)
        self.bi2 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.att = AttentionLayer(256)
        self.reg_dense = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(),
            nn.Linear(128, seq_len)
        )

    def forward(self, x):                     # x (B,1,L)
        cls = self.cls_head(self.cls_feat(x))     # (B,L)

        y = self.conv_reg(x).permute(0, 2, 1)     # (B,L,16)
        y,_ = self.bi1(y)
        y,_ = self.bi2(y)
        ctx, att = self.att(y)                    # (B,512)
        reg = self.reg_dense(ctx)                 # (B,L)

        return reg * cls, cls, att                # masked power, on/off, att


class RNN_attention_classification(Disaggregator):

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.MODEL_NAME = "RNN_attention_classification"
        self.chunk_wise_training = params.get("chunk_wise_training", True)
        self.sequence_length = params.get("sequence_length", 99)
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd")

        self.n_epochs   = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)

        self.appliance_params: Dict[str, Dict[str, float]] = {}
        self.mains_mean, self.mains_std = 1800, 600

        self.models: "OrderedDict[str,_RNNAttNet]" = OrderedDict()
        self.best: Dict[str, float] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fresh_network(self):
        return _RNNAttNet(self.sequence_length).to(self.device)
    
    def set_mains_params(self, mains_list):
        data = np.concatenate([m.values.flatten() for m in mains_list])
        self.mains_mean = data.mean()
        self.mains_std  = max(data.std(), 1.0)

    def set_appliance_params(self, train_apps):
        for app, dfs in train_apps:
            data = np.concatenate([d.values.flatten() for d in dfs])
            self.appliance_params[app] = {
                "mean": data.mean(),
                "std" : max(data.std(), 1.0),
                "min" : data.min(),
                "max" : data.max()
            }

    def classify(self, apps, threshold: float = 15.0):
        L, pad = self.sequence_length, self.sequence_length // 2
        out = []
        for app, dfs in apps:
            proc = []
            for df in dfs:
                v = df.values.flatten()
                v[v <= threshold] = 0
                v[v >  threshold] = 1
                v = np.pad(v, (pad, pad))
                w = np.array([v[i:i+L] for i in range(len(v)-L+1)], np.float32)
                proc.append(pd.DataFrame(w))
            out.append((app, proc))
        return out

    def partial_fit(self, mains, apps, do_preprocessing=True, **_):

        if not self.appliance_params:
            self.set_appliance_params(apps)
        self.set_mains_params(mains)

        if do_preprocessing:
            cls_targets = self.classify(copy.deepcopy(apps))
            mains, apps = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=mains,
                submeters_lst=apps,
                method="train",
                appliance_params=self.appliance_params,
                windowing=False
            )

        X = torch.tensor(pd.concat(mains).values,
                         dtype=torch.float32).unsqueeze(1)   # (N,1,L)
        N = X.size(0)
        perm = torch.randperm(N)
        split = int(0.15 * N)
        val_idx, tr_idx = perm[:split], perm[split:]
        X_tr, X_val = X[tr_idx].to(self.device), X[val_idx].to(self.device)

        y_reg, y_cls = {}, {}
        for app, dfs in apps:
            y_reg[app] = torch.tensor(pd.concat(dfs).values, dtype=torch.float32)
        for app, dfs in cls_targets:
            y_cls[app] = torch.tensor(pd.concat(dfs).values, dtype=torch.float32)

        mse, bce = nn.MSELoss(), nn.BCELoss()

        for app in y_reg:
            y_tr = y_reg[app][tr_idx].to(self.device)
            y_val = y_reg[app][val_idx].to(self.device)
            c_tr = y_cls[app][tr_idx].to(self.device)
            c_val = y_cls[app][val_idx].to(self.device)

            if app not in self.models:
                self.models[app] = self._fresh_network()
                self.best[app] = np.inf

            net = self.models[app]
            optim = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

            loader = DataLoader(
                TensorDataset(X_tr, y_tr, c_tr),
                batch_size=self.batch_size, shuffle=True
            )

            for ep in range(self.n_epochs):
                net.train()
                run_loss = 0.0
                bar = tqdm(loader,
                           desc=f"{app} ▏epoch {ep+1}/{self.n_epochs}",
                           leave=False, unit="batch")
                for xb, yb, cb in bar:
                    optim.zero_grad()
                    pr, pc, _ = net(xb)
                    loss = mse(pr, yb) + bce(pc, cb)
                    loss.backward()
                    optim.step()
                    run_loss += loss.item()
                    bar.set_postfix(loss=f"{loss.item():.4f}")

                avg_loss = run_loss / len(loader)

                net.eval()
                with torch.no_grad():
                    vr, vc, _ = net(X_val)
                    v_loss = mse(vr, y_val).item() + bce(vc, c_val).item()

                tqdm.write(
                    f"[{app}] Epoch {ep+1}/{self.n_epochs} | "
                    f"Train Loss: {avg_loss:.4f} | Val Loss: {v_loss:.4f}"
                )

                if v_loss < self.best[app]:
                    self.best[app] = v_loss
                    torch.save(net.state_dict(), f"rnn_att-{app}.pth")

            net.load_state_dict(torch.load(f"rnn_att-{app}.pth",
                                           map_location=self.device))

    def disaggregate_chunk(self, mains, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model
        if do_preprocessing:
            mains = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=mains,
                submeters_lst=None,
                method="test",
                appliance_params=self.appliance_params,
                windowing=False
            )

        L = self.sequence_length
        out = []
        for m in mains:
            X = torch.tensor(m.values, dtype=torch.float32
                            ).unsqueeze(1).to(self.device)
            disc = {}
            for app, net in self.models.items():
                net.eval()
                with torch.no_grad():
                    pr, _, _ = net(X)
                    pr = pr.cpu().numpy()

                # overlap-mean
                def ov(a):
                    s, c = np.zeros(len(a)+L-1), np.zeros(len(a)+L-1)
                    for i,row in enumerate(a):
                        s[i:i+L] += row
                        c[i:i+L] += 1
                    return s/c

                power = ov(pr)
                p = self.appliance_params[app]
                power = np.clip(p["min"] + power*(p["max"]-p["min"]), 0, None)
                disc[app] = pd.Series(power, dtype="float32")
            out.append(pd.DataFrame(disc, dtype="float32"))
        return out

    # NILMTK shortcut wrappers
    def train(self, mains, apps, **kw):
        return self.partial_fit(mains, apps, **kw)

    def disaggregate(self, mains, store):
        preds = self.disaggregate_chunk(mains)
        for i, df in enumerate(preds):
            for col in df.columns:
                store.put(f"/building1/elec/meter{i+1}/{col}", df[col])
