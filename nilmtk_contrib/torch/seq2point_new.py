from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from nilmtk.disaggregate import Disaggregator


class SequenceLengthError(Exception):
    pass


class ApplianceNotFoundError(Exception):
    pass


class Seq2PointTorch(Disaggregator):
    def __init__(self, params):
        super().__init__()
        self.MODEL_NAME = "Seq2PointTorch"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        self.mains_mean = params.get("mains_mean", 1800)
        self.mains_std = params.get("mains_std", 600)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length should be odd!")

    def _build_network(self):
        seq_len = self.sequence_length
        conv_reduction = (10-1) + (8-1) + (6-1) + (5-1) + (5-1)  # = 29
        model = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, stride=1), nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, stride=1), nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, stride=1), nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, stride=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(50, 50, kernel_size=5, stride=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(50 * (seq_len - conv_reduction), 1024), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1)
        )
        return model.to(self.device)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True,
                    current_epoch=0, **load_kwargs):
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, "train"
            )

        train_main = pd.concat(train_main, axis=0).values.reshape(
            -1, self.sequence_length, 1
        )
        train_main = torch.tensor(train_main, dtype=torch.float32).permute(0, 2, 1)

        new_train_apps = []
        for app_name, app_df_list in train_appliances:
            app_df = pd.concat(app_df_list, axis=0).values.reshape(-1, 1)
            new_train_apps.append(
                (app_name, torch.tensor(app_df, dtype=torch.float32))
            )
        train_appliances = new_train_apps

        n_total = train_main.size(0)
        val_split = int(0.15 * n_total)
        idx = torch.randperm(n_total)
        tr_idx, val_idx = idx[val_split:], idx[:val_split]

        mains_train = train_main[tr_idx].to(self.device)
        mains_val = train_main[val_idx].to(self.device)

        for appliance, power_tensor in train_appliances:
            power_tensor = power_tensor.to(self.device)
            power_train = power_tensor[tr_idx]
            power_val = power_tensor[val_idx]

            if appliance not in self.models:
                print("First model training for", appliance)
                self.models[appliance] = self._build_network()
            else:
                print("Started Retraining model for", appliance)

            model = self.models[appliance]
            optimiser = torch.optim.Adam(model.parameters())
            loss_fn = nn.MSELoss()

            best_val = np.inf
            best_file = f"{self.file_prefix}-{appliance.replace(' ', '_')}-epoch{current_epoch}.pth"

            dataset = TensorDataset(mains_train, power_train)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.n_epochs):
                model.train()
                epoch_losses = []

                for x_batch, y_batch in loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    optimiser.zero_grad()
                    preds = model(x_batch).squeeze(1)
                    loss = loss_fn(preds, y_batch)
                    loss.backward()
                    optimiser.step()
                    epoch_losses.append(loss.item())

                model.eval()
                with torch.no_grad():
                    val_preds = model(mains_val).squeeze(1)
                    val_loss = loss_fn(val_preds, power_val).item()

                avg_loss = np.mean(epoch_losses)
                tqdm.write(f"[{appliance}] Epoch {epoch+1}/{self.n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), best_file)

            model.load_state_dict(torch.load(best_file, map_location=self.device))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method="test"
            )

        results = []
        for mains_df in test_main_list:
            mains_np = mains_df.values.reshape(-1, self.sequence_length, 1)
            mains_tensor = (
                torch.tensor(mains_np, dtype=torch.float32)
                .permute(0, 2, 1)
                .to(self.device)
            )

            disagg = {}
            for appliance, net in self.models.items():
                net.eval()
                with torch.no_grad():
                    preds = (
                        net(mains_tensor).cpu().numpy().flatten()
                        * self.appliance_params[appliance]["std"]
                        + self.appliance_params[appliance]["mean"]
                    )
                    preds = np.clip(preds, 0, None)
                    disagg[appliance] = pd.Series(preds, dtype="float32")

            results.append(pd.DataFrame(disagg, dtype="float32"))
        return results

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == "train":
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                pad = n // 2
                new_mains = np.pad(new_mains, (pad, pad), "constant")
                new_mains = np.array(
                    [new_mains[i: i + n] for i in range(len(new_mains) - n + 1)]
                )
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_name, app_df_list in submeters_lst:
                if app_name not in self.appliance_params:
                    raise ApplianceNotFoundError(
                        f"Parameters for {app_name} were not found!"
                    )
                app_mean = self.appliance_params[app_name]["mean"]
                app_std = self.appliance_params[app_name]["std"]

                processed = []
                for app_df in app_df_list:
                    vals = app_df.values.reshape(-1, 1)
                    vals = (vals - app_mean) / app_std
                    processed.append(pd.DataFrame(vals))
                appliance_list.append((app_name, processed))
            return mains_df_list, appliance_list

        mains_df_list = []
        for mains in mains_lst:
            new_mains = mains.values.flatten()
            n = self.sequence_length
            pad = n // 2
            new_mains = np.pad(new_mains, (pad, pad), "constant")
            new_mains = np.array(
                [new_mains[i: i + n] for i in range(len(new_mains) - n + 1)]
            )
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            mains_df_list.append(pd.DataFrame(new_mains))
        return mains_df_list

    def set_appliance_params(self, train_appliances):
        for app_name, df_list in train_appliances:
            data = np.concatenate([df.values.flatten() for df in df_list])
            mean, std = data.mean(), data.std()
            if std < 1:
                std = 100
            self.appliance_params[app_name] = {"mean": mean, "std": std}
        print(self.appliance_params)
