import os, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from nilmtk.disaggregate import Disaggregator
from nilmtk_contrib.torch.preprocessing import preprocess

class Seq2SeqModel(nn.Module):
    """
    Sequence-to-Sequence CNN model that maps input power sequences 
    to output appliance power sequences of the same length.
    """
    def __init__(self, seq_len):
        super().__init__()

        self.seq_len = seq_len
        
        # Encoder: 1D CNN layers with different strides for feature extraction
        self.conv1 = nn.Conv1d(1, 30, 10, stride=2)
        self.conv2 = nn.Conv1d(30,30, 8,  stride=2)
        self.conv3 = nn.Conv1d(30,40, 6,  stride=1)
        self.conv4 = nn.Conv1d(40,50, 5,  stride=1)
        self.dropout1 = nn.Dropout(.2)
        self.conv5 = nn.Conv1d(50,50, 5, stride=1)
        self.dropout2 = nn.Dropout(.2)

        # Calculate the flattened size after all convolutions
        L = seq_len
        L = (L - 10)//2 + 1
        L = (L - 8)//2 + 1
        L = L - 6 + 1
        L = L - 5 + 1
        L = L - 5 + 1
        flat_size = 50 * L

        # Decoder: Fully connected layers to reconstruct sequence
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(flat_size, 1024)
        self.dropout3 = nn.Dropout(.2)
        self.fc2      = nn.Linear(1024, seq_len)  # Output same length as input

    def forward(self, x):
        # Input: [B, seq_len, 1] → rearrange for Conv1d: [B, 1, seq_len]
        x = x.permute(0,2,1)
        
        # Encoder: feature extraction through conv layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv5(x))
        x = self.dropout2(x)
        
        # Decoder: reconstruct to original sequence length
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)           # [B, seq_len]
        return x

class Seq2Seq(Disaggregator):
    """
    NILM disaggregator using sequence-to-sequence learning.
    Maps input power sequences to appliance power sequences of the same length.
    """
    def __init__(self, params):
        super().__init__()

        self.MODEL_NAME = "Seq2Seq"
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Extract hyperparameters
        self.sequence_length     = params.get('sequence_length', 99)
        if self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be odd")
        self.n_epochs            = params.get('n_epochs', 10)
        self.batch_size          = params.get('batch_size', 512)
        self.mains_mean          = 1800
        self.mains_std           = 600
        self.appliance_params    = params.get('appliance_params', {})  # Normalization stats
        self.models              = OrderedDict()  # Store separate models for each appliance
        self.device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def return_network(self):
        """Factory method to create a new Seq2Seq model instance"""
        return Seq2SeqModel(self.sequence_length).to(self.device)

    def set_appliance_params(self, train_appliances):
        """Compute normalization statistics (mean, std) for each appliance"""
        for name, lst in train_appliances:
            arr = pd.concat(lst, axis=0).values.flatten()
            m, s = arr.mean(), arr.std()
            # Prevent division by zero in normalization
            if s < 1: s = 100
            self.appliance_params[name] = {'mean':m, 'std':s}

    def partial_fit(self, train_main, train_appliances,
                    do_preprocessing=True, current_epoch=0, **_):
        """Train models on a chunk of data (supports incremental learning)"""
        
        # Compute appliance-specific normalization parameters if not provided
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        # Preprocess data: windowing, normalization, etc.
        if do_preprocessing:
            train_main, train_appliances = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=train_main,
                submeters_lst=train_appliances,
                method="train",
                appliance_params=self.appliance_params,
                windowing=True
            )

        # Prepare main power data for training
        mains_arr = pd.concat(train_main,axis=0).values \
                     .reshape(-1, self.sequence_length, 1)

        # Train a separate model for each appliance
        for name, dfs in train_appliances:
            # Prepare appliance power sequences (targets)
            arr = pd.concat(dfs,axis=0).values \
                    .reshape(-1, self.sequence_length)
            
            # Create new model if this appliance hasn't been seen before
            if name not in self.models:
                self.models[name] = self.return_network()
            model = self.models[name]

            # Convert to tensors
            X = torch.tensor(mains_arr, dtype=torch.float32)
            Y = torch.tensor(arr,       dtype=torch.float32)
            
            # Split into training and validation sets
            split = int(0.85*len(X))

            tr_ds = TensorDataset(X[:split], Y[:split])
            va_ds = TensorDataset(X[split:], Y[split:])
            tr = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
            va = DataLoader(va_ds, batch_size=self.batch_size)

            # Setup training components
            opt     = optim.Adam(model.parameters())
            loss_fn = nn.MSELoss()
            best    = float('inf')
            ckpt    = f"{self.file_prefix}-{name}-epoch{current_epoch}.pt"

            # Training loop
            for epoch in tqdm(range(self.n_epochs), desc=f"Train {name}"):
                # Training phase
                model.train()
                for xb, yb in tr:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    out = model(xb)                   # [B, seq_len]
                    loss_fn(out, yb).backward()
                    opt.step()

                # Validation phase
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in va:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        val_losses.append(loss_fn(model(xb), yb).item())
                val_loss = sum(val_losses)/len(val_losses)
                
                # Save best model based on validation loss
                if val_loss < best:
                    best = val_loss
                    torch.save(model.state_dict(), ckpt)

            # Load the best model weights
            model.load_state_dict(torch.load(ckpt, map_location=self.device))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregate power consumption using overlapping windows and averaging"""
        
        if model: self.models = model
        
        # Preprocess test data similar to training data
        if do_preprocessing:
            test_main_list = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=test_main_list,
                submeters_lst=None,
                method="test",
                appliance_params=self.appliance_params,
                windowing=True
            )

        results = []
        n = self.sequence_length
        
        # Process each chunk of test data
        for tm in test_main_list:
            arr = tm.values.reshape(-1, n)
            ds  = DataLoader(TensorDataset(torch.tensor(arr, dtype=torch.float32)),
                             batch_size=self.batch_size)
            outd = {}
            
            # Get predictions from each appliance model
            for name, m in self.models.items():
                preds = []
                m.eval()
                with torch.no_grad():
                    for (xb_cpu,) in ds:
                        # Unsqueeze back to [B, seq_len, 1] for model input
                        xb = xb_cpu.unsqueeze(-1).to(self.device)
                        p  = m(xb).cpu().numpy()    # [B, seq_len]
                        preds.append(p)
                
                # Concatenate all predictions
                P = np.concatenate(preds, axis=0)
                
                # Reconstruct full sequence by averaging overlapping windows
                total = P.shape[0] + n - 1
                sum_arr    = np.zeros(total)
                counts_arr = np.zeros(total)
                for i in range(P.shape[0]):
                    sum_arr[i:i+n]    += P[i]
                    counts_arr[i:i+n] += 1
                avg = sum_arr/counts_arr
                
                # Denormalize predictions back to original power scale
                mpar = self.appliance_params[name]
                out  = mpar['mean'] + avg * mpar['std']
                
                # Ensure non-negative power values
                outd[name] = pd.Series(np.clip(out, 0, None))
                
            # Combine all appliance predictions for this chunk
            results.append(pd.DataFrame(outd, dtype='float32'))
        return results