import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from nilmtk.disaggregate import Disaggregator
from nilmtk_contrib.utils.checkpoints import (
    build_metadata,
    collect_dependencies,
    load_metadata,
    load_torch_state,
    save_metadata,
    save_torch_state,
    temporary_checkpoint,
)
from nilmtk_contrib.utils.logging import get_logger
from nilmtk_contrib.utils.model import initialize_runtime, legacy_print
from nilmtk_contrib.utils.params import normalize_common_params
from nilmtk_contrib.utils.random import set_random_seed
from nilmtk_contrib.utils.validation import train_validation_split

logger = get_logger(__name__)
_log_print = legacy_print(logger)

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
    """
    Denoising Autoencoder for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation"
    https://arxiv.org/abs/1507.06594
    
    The model uses a denoising autoencoder architecture for energy disaggregation tasks,
    learning to reconstruct individual appliance power consumption from aggregate
    household power measurements.
    
    Architecture Overview:
    - Convolutional encoder layer for feature extraction
    - Fully connected bottleneck layers for dimensionality reduction
    - Convolutional decoder layer for sequence reconstruction
    - Sequence-to-sequence prediction for energy disaggregation
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 99)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - mains_mean (float): Mean value for mains normalization (default: 1000)
            - mains_std (float): Standard deviation for mains normalization (default: 600)
            - appliance_params (dict): Appliance-specific normalization parameters
            - save-model-path (str): Path to save trained models
            - pretrained-model-path (str): Path to load pre-trained models
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        super().__init__()
        common = normalize_common_params(
            params,
            defaults={
                "sequence_length": 99,
                "n_epochs": 10,
                "batch_size": 512,
                "mains_mean": 1000,
                "mains_std": 600,
                "appliance_params": {},
                "save_model_path": None,
                "pretrained_model_path": None,
                "chunk_wise_training": False,
                "seed": None,
                "verbose": False,
                "device": None,
            },
        )
        self.MODEL_NAME        = "DAE"
        self.file_prefix       = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.sequence_length   = common.sequence_length
        self.n_epochs          = common.n_epochs
        self.batch_size        = common.batch_size
        self.mains_mean        = common.mains_mean
        self.mains_std         = common.mains_std
        self.appliance_params  = common.appliance_params
        self.save_model_path   = common.save_model_path
        self.load_model_path   = common.pretrained_model_path
        self.chunk_wise_training = common.chunk_wise_training
        self.seed              = common.seed
        self.verbose           = common.verbose
        self.models            = OrderedDict()
        device = common.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device            = torch.device(device)
        set_random_seed(self.seed, backends=("python", "numpy", "torch"))
        if self.load_model_path:
            self.load_model()

    def return_network(self):
        """Returns the DAE model."""
        return DAEModel(self.sequence_length).to(self.device)

    def set_appliance_params(self, train_appliances):
        """
        Set the mean and std for each appliance based on the training data.
        """
        for name, lst in train_appliances:
            arr = pd.concat(lst, axis=0).values.flatten()
            m, s = arr.mean(), arr.std()
            if s < 1:
                s = 100  # avoid zero std
            self.appliance_params[name] = {'mean': m, 'std': s}

    def normalize_input(self, data, n, mean, std, overlap):
        """
        Normalizes and windows the input data.
        """
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
        """
        Denormalizes the output data.
        """
        return mean + data*std

    def call_preprocessing(self, mains_lst, subs, method):
        """
        Preprocesses the mains and appliance data.
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
        """
        Trains the model on a chunk of data.
        """
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
            split = train_validation_split(
                X,
                Y,
                validation_fraction=0.15,
                strategy="tail",
                min_train=1,
                min_val=1,
                allow_no_validation=True,
            )
            if not split.metadata.should_train:
                continue

            tr_ds = TensorDataset(split.X_train, split.y_train)  # train set
            tr = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)  # train loader
            va = None
            if split.metadata.validation_enabled:
                va_ds = TensorDataset(split.X_val, split.y_val)  # validation set
                va = DataLoader(va_ds, batch_size=self.batch_size)  # validation loader

            opt     = optim.Adam(model.parameters())
            loss_fn = nn.MSELoss()
            best = float('inf')
            with temporary_checkpoint(".pt") as ckpt:
                epochs = tqdm(range(self.n_epochs), desc=name, disable=not self.verbose)
                for _ in epochs:
                    model.train()
                    for xb, yb in tr:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        opt.zero_grad()
                        out = model(xb)
                        loss_fn(out, yb).backward()
                        opt.step()

                    if va is None:
                        save_torch_state(model, ckpt)
                    else:
                        model.eval()
                        vl = []
                        with torch.no_grad():
                            for xb, yb in va:
                                xb, yb = xb.to(self.device), yb.to(self.device)
                                vl.append(loss_fn(model(xb), yb).item())
                        if vl:
                            val_loss = sum(vl)/len(vl)
                            if val_loss < best:
                                best = val_loss
                                save_torch_state(model, ckpt)

                if ckpt.exists():
                    load_torch_state(model, ckpt, self.device)

        if self.save_model_path:
            self.save_model()

    def save_model(self):
        """
        Saves the trained model and parameters.
        """
        model_folder = Path(self.save_model_path)
        model_folder.mkdir(parents=True, exist_ok=True)
        metadata = build_metadata(
            model_class=self.MODEL_NAME,
            backend="torch",
            sequence_length=self.sequence_length,
            appliance_params=self.appliance_params,
            mains_mean=self.mains_mean,
            mains_std=self.mains_std,
            dependencies=collect_dependencies(["nilmtk-contrib", "torch", "numpy", "pandas"]),
        )
        save_metadata(model_folder, metadata)
        for name, m in self.models.items():
            logger.info("Saving %s model for %s.", self.MODEL_NAME, name)
            save_torch_state(m, model_folder / f"{name}.pt")

    def load_model(self):
        """
        Loads a pre-trained model and its parameters.
        """
        model_folder = Path(self.load_model_path)
        metadata_path = model_folder / "metadata.json"
        if metadata_path.exists():
            p = load_metadata(
                model_folder,
                expected_model_class=self.MODEL_NAME,
                expected_backend="torch",
            )
        else:
            logger.warning(
                "Loading legacy %s model metadata from model.json.", self.MODEL_NAME
            )
            with open(model_folder / 'model.json') as f:
                p = json.load(f)
        self.sequence_length = p['sequence_length']
        self.mains_mean      = p['mains_mean']
        self.mains_std       = p['mains_std']
        self.appliance_params= p['appliance_params']
        for name in self.appliance_params:
            m = self.return_network()
            load_torch_state(m, model_folder / f"{name}.pt", self.device)
            self.models[name] = m

    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        """
        Disaggregates a chunk of mains data.
        """
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
                mean = self.appliance_params[name]["mean"]
                std = self.appliance_params[name]["std"]
                p_den = self.denormalize_output(p_all, mean, std).flatten()
                p_den = np.clip(p_den, 0, None)
                outd[name] = pd.Series(p_den)
            results.append(pd.DataFrame(outd, dtype='float32'))
        return results
