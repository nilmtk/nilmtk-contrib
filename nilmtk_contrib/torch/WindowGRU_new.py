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
    """
    Neural network combining 1D CNN feature extraction with bidirectional GRU layers
    for sequence-to-point NILM disaggregation.
    """
    def __init__(self, sequence_length):
        super(GRUNet, self).__init__()
        # 1D CNN for initial feature extraction
        self.conv1    = nn.Conv1d(1, 16, kernel_size=4, padding=2)
        
        # Bidirectional GRU layers for sequence modeling
        self.gru1     = nn.GRU(16, 64, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2     = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # Final layers for single value prediction
        self.fc1      = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(128, 1)

    def forward(self, x):
        # Extract features using 1D convolution
        x = self.conv1(x)           # [batch, 1, seq_len] -> [batch, 16, seq_len]
        x = torch.relu(x)
        x = x.permute(0, 2, 1)      # Rearrange for GRU: [batch, seq_len, 16]
        
        # Process through bidirectional GRU layers
        x, _   = self.gru1(x)       # [batch, seq_len, 128]
        x      = self.dropout1(x)
        _, h_n = self.gru2(x)       # h_n: [2, batch, 128] (final hidden states)
        
        # Combine forward and backward final states
        h      = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, 256]
        h      = self.dropout2(h)
        
        # Final prediction layers
        h      = self.fc1(h)        # [batch, 128]
        h      = torch.relu(h)
        h      = self.dropout3(h)
        out    = self.fc2(h)        # [batch, 1]
        return out

class WindowGRU(Disaggregator):
    """
    NILM disaggregator using windowed GRU approach with custom preprocessing.
    Uses sliding windows and GRU networks for appliance disaggregation.
    """
    def __init__(self, params):
        super().__init__()
        self.MODEL_NAME      = "WindowGRU"
        self.file_prefix     = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Extract hyperparameters
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path', None)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs        = params.get('n_epochs', 10)
        self.batch_size      = params.get('batch_size', 512)
        self.max_val         = 800  # Normalization factor
        self.models          = OrderedDict()  # Store separate models for each appliance
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def return_network(self):
        """Factory method to create a new GRU model instance"""
        return GRUNet(self.sequence_length).to(self.device)

    def partial_fit(self, train_main, train_appliances,
                    do_preprocessing=True, current_epoch=0, **kwargs):
        """Train models on a chunk of data (supports incremental learning)"""
        
        # Preprocess data using custom windowing approach
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train'
            )

        # Prepare main power data for training
        mains_arr = pd.concat(train_main, axis=0).values \
                    .reshape(-1, self.sequence_length)  # [N, seq_len]
        
        # Prepare appliance power data 
        new_apps = []
        for app_name, df_list in train_appliances:
            concatenated = pd.concat(df_list, axis=0)
            arr = concatenated.values.reshape(-1, 1)      # [N, 1]
            new_apps.append((app_name, arr))

        # Train a separate model for each appliance
        for app_name, arr in new_apps:
            # Create new model if this appliance hasn't been seen before
            if app_name not in self.models:
                self.models[app_name] = self.return_network()
            model = self.models[app_name]

            # Convert to tensors and split into train/validation
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

            # Setup training components
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            best_val  = float('inf')
            ckpt_path = f"{self.file_prefix}-{app_name.replace(' ','_')}-epoch{current_epoch}.pt"

            # Training loop
            for epoch in tqdm(range(self.n_epochs),
                              desc=f"Train {app_name}"):
                # Training phase
                model.train()
                for xb_cpu, yb_cpu in train_loader:
                    xb = xb_cpu.unsqueeze(1).to(self.device)  # Add channel dim: [B,1,seq]
                    yb = yb_cpu.to(self.device)               # [B,1]
                    optimizer.zero_grad()
                    out = model(xb)                           # [B,1]
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    
                # Validation phase
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb_cpu, yb_cpu in val_loader:
                        xb = xb_cpu.unsqueeze(1).to(self.device)
                        yb = yb_cpu.to(self.device)
                        out = model(xb)
                        val_losses.append(criterion(out, yb).item())
                val_loss = sum(val_losses) / len(val_losses)
                
                # Save best model based on validation loss
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), ckpt_path)
                    
            # Load the best model weights
            model.load_state_dict(torch.load(ckpt_path,
                                             map_location=self.device))
            torch.cuda.empty_cache()

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregate power consumption for each appliance from aggregate mains data"""
        
        if model is not None:
            self.models = model
            
        # Preprocess test data using custom windowing
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, None, 'test'
            )

        results = []
        
        # Process each chunk of test data
        for mains in test_main_list:
            arr = mains.values.reshape(-1, self.sequence_length)
            x_cpu = torch.tensor(arr, dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(x_cpu),
                                     batch_size=self.batch_size)
            out_dict = {}
            
            # Get predictions from each appliance model
            for app_name, m in self.models.items():
                preds = []
                m.eval()
                with torch.no_grad():
                    for (xb_cpu,) in test_loader:
                        xb = xb_cpu.unsqueeze(1).to(self.device)
                        p  = m(xb).view(-1).cpu().numpy()
                        preds.append(p)
                        
                # Combine predictions and denormalize
                all_pred = np.concatenate(preds)
                all_pred = np.clip(all_pred, 0, None) * self.max_val
                out_dict[app_name] = pd.Series(all_pred)
                torch.cuda.empty_cache()
                
            # Combine all appliance predictions for this chunk
            results.append(pd.DataFrame(out_dict, dtype='float32'))
        return results

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """Custom preprocessing with sliding window approach"""
        
        if method == 'train':
            pm, apps = [], []
            
            # Process mains data with padding and windowing
            for mains in mains_lst:
                pad = [0] * (self.sequence_length - 1)
                tmp = pd.concat([mains,
                                 pd.DataFrame({mains.columns[0]: pad})])
                pm.append(pd.DataFrame(self.preprocess_train_mains(tmp)))
                
            # Process appliance data
            for name, lst in submeters_lst:
                dfs = [pd.DataFrame(self.preprocess_train_appliances(df))
                       for df in lst]
                apps.append((name, dfs))
            return pm, apps

        if method == 'test':
            pm = []
            
            # Process test mains data with padding and windowing
            for mains in mains_lst:
                pad = [0] * (self.sequence_length - 1)
                tmp = pd.concat([mains,
                                 pd.DataFrame({mains.columns[0]: pad})])
                pm.append(pd.DataFrame(self.preprocess_test_mains(tmp)))
            return pm

    def preprocess_train_mains(self, mains):
        """Create sliding windows from mains data for training"""
        arr = (mains / self.max_val).values
        # Create sliding window indices
        idx = (np.arange(self.sequence_length)[None, :]
               + np.arange(len(arr) - self.sequence_length + 1)[:, None])
        return arr[idx].reshape(-1, self.sequence_length)

    def preprocess_train_appliances(self, app):
        """Normalize appliance data for training"""
        return (app / self.max_val).values.reshape(-1, 1)

    def preprocess_test_mains(self, mains):
        """Create sliding windows from mains data for testing"""
        arr = (mains / self.max_val).values
        # Create sliding window indices
        idx = (np.arange(self.sequence_length)[None, :]
               + np.arange(len(arr) - self.sequence_length + 1)[:, None])
        return arr[idx].reshape(-1, self.sequence_length)

    def _normalize(self, chunk, m):
        """Normalize data by dividing by maximum value"""
        return chunk / m

    def _denormalize(self, chunk, m):
        """Denormalize data by multiplying by maximum value"""
        return chunk * m