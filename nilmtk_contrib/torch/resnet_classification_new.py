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
        self.relu = nn.ReLU()

    def forward(self, x):
        s = x
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.c3(x)
        return self.relu(x + s)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, mid: int, out: int, k: int):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, mid, k, padding="same")
        self.c2 = nn.Conv1d(mid,   mid, k, padding="same")
        self.c3 = nn.Conv1d(mid,   out, k, padding="same")
        self.proj = nn.Conv1d(in_ch, out, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        s = self.proj(x)
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.c3(x)
        return self.relu(x + s)


class _ResNetNet(nn.Module):
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
        self.cls_head = nn.Linear(1024, seq_len)

        self.pad   = nn.ConstantPad1d((3, 3), 0)
        self.conv0 = nn.Conv1d(1, 30, 48, stride=2)
        self.bn0   = nn.BatchNorm1d(30)
        self.pool0 = nn.MaxPool1d(3, stride=2)
        self.block1 = ConvBlock(30, 30, 30, 24)
        self.block2 = IdentityBlock(30, 12)
        self.block3 = IdentityBlock(30,  6)
        self.reg_end = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, seq_len)
        )

    def forward(self, x):
        cls = torch.sigmoid(self.cls_head(self.cls_feat(x)))
        y   = self.pad(x)
        y   = F.relu(self.bn0(self.conv0(y)))
        y   = self.pool0(y)
        y   = self.block1(y)
        y   = self.block2(y)
        y   = self.block3(y)
        reg = self.reg_end(y)
        return reg * cls, cls


class ResNet_classification(Disaggregator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.MODEL_NAME = "ResNet_classification"
        self.chunk_wise_training = params.get("chunk_wise_training", True)
        self.sequence_length = params.get("sequence_length", 99)
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("sequence_length must be odd")

        self.n_epochs   = params.get("n_epochs",   10)
        self.batch_size = params.get("batch_size", 512)

        self.mains_mean, self.mains_std = 1800, 600
        self.appliance_params: Dict[str, Dict[str, float]] = {}

        self.models: "OrderedDict[str,_ResNetNet]" = OrderedDict()
        self.optims:  Dict[str, torch.optim.Optimizer] = {}
        self.best:    Dict[str, float] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def partial_fit(self, mains, appliances, do_preprocessing=True, **_):
        if not self.appliance_params:
            self.set_appliance_params(appliances)
        self._set_mains_params(mains)

        if do_preprocessing:
            cls_labels = self._make_on_off(copy.deepcopy(appliances))
            mains, appliances = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=mains,
                submeters_lst=appliances,
                method="train",
                appliance_params=self.appliance_params,
                windowing=False
            )

        X = torch.tensor(pd.concat(mains).values, dtype=torch.float32).unsqueeze(1)
        N = X.size(0)
        perm = torch.randperm(N)
        val_idx, tr_idx = perm[:int(0.15 * N)], perm[int(0.15 * N):]
        X_tr, X_val = X[tr_idx].to(self.device), X[val_idx].to(self.device)

        y_reg, y_cls = {}, {}
        for app, dfs in appliances:
            y_reg[app] = torch.tensor(pd.concat(dfs).values, dtype=torch.float32)
        for app, dfs in cls_labels:
            y_cls[app] = torch.tensor(pd.concat(dfs).values, dtype=torch.float32)

        mse, bce = nn.MSELoss(), nn.BCELoss()

        for app in y_reg:
            y_tr = y_reg[app][tr_idx].to(self.device)
            y_val = y_reg[app][val_idx].to(self.device)
            c_tr = y_cls[app][tr_idx].to(self.device)
            c_val = y_cls[app][val_idx].to(self.device)

            if app not in self.models:
                net = _ResNetNet(self.sequence_length).to(self.device)
                self.models[app] = net
                self.optims[app] = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
                self.best[app]   = np.inf

            net, opt = self.models[app], self.optims[app]
            loader = DataLoader(TensorDataset(X_tr, y_tr, c_tr),
                                batch_size=self.batch_size, shuffle=True)

            for ep in range(self.n_epochs):
                net.train()
                ep_bar = tqdm(loader,
                              desc=f"{app} ▏epoch {ep+1}/{self.n_epochs}",
                              unit="batch", leave=False)   # live bar
                running = 0.0
                for xb, yb, cb in ep_bar:
                    opt.zero_grad()
                    pr, pc = net(xb)
                    loss = mse(pr, yb) + bce(pc, cb)
                    loss.backward()
                    opt.step()
                    running += loss.item()
                    ep_bar.set_postfix(loss=f"{loss.item():.4f}")  # update

                avg_loss = running / len(loader)

                # validation
                net.eval()
                with torch.no_grad():
                    vr, vc = net(X_val)
                    v_loss = mse(vr, y_val).item() + bce(vc, c_val).item()

                tqdm.write(f"[{app}] Epoch {ep+1}/{self.n_epochs} | " f"Train Loss: {avg_loss:.4f} | Val Loss: {v_loss:.4f}")   

                if v_loss < self.best[app]:
                    self.best[app] = v_loss
                    torch.save(net.state_dict(), f"resnet_cls-{app}.pth")

            net.load_state_dict(torch.load(f"resnet_cls-{app}.pth", map_location=self.device))

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
            X = torch.tensor(m.values, dtype=torch.float32).unsqueeze(1).to(self.device)
            disc = {}
            for app, net in self.models.items():
                net.eval()
                with torch.no_grad():
                    pr, _ = net(X)
                    pr = pr.cpu().numpy()

                def overlap(wins):
                    s, c = np.zeros(len(wins)+L-1), np.zeros(len(wins)+L-1)
                    for i in range(len(wins)):
                        s[i:i+L] += wins[i].flatten()
                        c[i:i+L] += 1
                    return s / c

                power = overlap(pr)
                p = self.appliance_params[app]
                power = np.clip(p["min"] + power*(p["max"]-p["min"]), 0, None)
                disc[app] = pd.Series(power, dtype="float32")
            out.append(pd.DataFrame(disc, dtype="float32"))
        return out

    def _make_on_off(self, apps):
        TH, n, pad = 15, self.sequence_length, self.sequence_length//2
        res = []
        for app, dfs in apps:
            lbls = []
            for df in dfs:
                a = df.copy()
                a[a<=TH] = 0; a[a>TH] = 1
                v = np.pad(a.values.flatten(), (pad,pad))
                w = np.array([v[i:i+n] for i in range(len(v)-n+1)])
                lbls.append(pd.DataFrame(w))
            res.append((app, lbls))
        return res

    def set_appliance_params(self, apps):
        for app, dfs in apps:
            data = np.concatenate([d.values.flatten() for d in dfs])
            self.appliance_params[app] = {
                "mean": data.mean(),
                "std":  max(data.std(), 1.0),
                "min":  data.min(),
                "max":  data.max()
            }

    def _set_mains_params(self, mains):
        data = np.concatenate([m.values.flatten() for m in mains])
        self.mains_mean, self.mains_std = data.mean(), data.std()

    # NILMTK wrappers
    def train(self, mains, apps, **kw):
        return self.partial_fit(mains, apps, **kw)

    def disaggregate(self, mains, store):
        preds = self.disaggregate_chunk(mains)
        for i, df in enumerate(preds):
            for col in df.columns:
                store.put(f"/building1/elec/meter{i+1}/{col}", df[col])
