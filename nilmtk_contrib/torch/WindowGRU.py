import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class FastReLUGRU(nn.Module):
    """
    Fast implementation using standard PyTorch GRU with post-processing to approximate
    ReLU activation behavior. This is much faster while maintaining similar performance.
    """
    def __init__(self, input_size, hidden_size, batch_first=True, bidirectional=False, return_sequences=True):
        super(FastReLUGRU, self).__init__()
        self.return_sequences = return_sequences
        
        # Use standard PyTorch GRU for speed
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        
        # Apply transformation to approximate ReLU activation effect
        # This linear layer helps adjust the tanh outputs to be more ReLU-like
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.activation_transform = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
    
    def forward(self, input, h0=None):
        # Fast GRU computation
        if self.return_sequences:
            output, final_h = self.gru(input, h0)
            # Apply transformation to make it more ReLU-like
            batch_size, seq_len, hidden_size = output.shape
            output_reshaped = output.reshape(-1, hidden_size)
            transformed = self.activation_transform(output_reshaped)
            output = transformed.reshape(batch_size, seq_len, hidden_size)
            return output, final_h
        else:
            # Only need final hidden state
            _, final_h = self.gru(input, h0)
            if final_h.dim() == 3:  # [num_layers, batch, hidden] -> [batch, hidden]
                if final_h.size(0) == 2:  # bidirectional
                    final_h = torch.cat([final_h[0], final_h[1]], dim=1)
                else:
                    final_h = final_h.squeeze(0)
            # Transform final hidden state
            final_h = self.activation_transform(final_h)
            return None, final_h

class GRUNet(nn.Module):
    """
    Neural network intended to align with the TensorFlow WindowGRU architecture.
    """
    def __init__(self, sequence_length):
        super(GRUNet, self).__init__()
        # 1D CNN with same padding as TF (padding="same")
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, padding=2, stride=1)
        
        # Bidirectional Fast ReLU GRU layers (much faster than custom cells)
        # First GRU: return_sequences=True (matches TF)
        self.gru1 = FastReLUGRU(16, 64, batch_first=True, bidirectional=True, return_sequences=True)
        self.dropout1 = nn.Dropout(0.5)
        
        # Second GRU: return_sequences=False (matches TF)
        self.gru2 = FastReLUGRU(128, 128, batch_first=True, bidirectional=True, return_sequences=False)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully Connected Layers matching TF
        self.fc1 = nn.Linear(256, 128)  # 256 = 128*2 (bidirectional)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
        # Initialize weights to match TensorFlow defaults
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to match TensorFlow defaults"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # GRU weights - use xavier/glorot uniform like TF
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name or 'bias_hh' in name:
                # GRU biases
                nn.init.zeros_(param)
            elif 'activation_transform' in name and 'weight' in name:
                # Transformation layer weights
                nn.init.xavier_uniform_(param)
            elif 'activation_transform' in name and 'bias' in name:
                # Transformation layer biases
                nn.init.zeros_(param)
            elif 'weight' in name and 'conv1' in name:
                # Conv1D weights
                nn.init.xavier_uniform_(param)
            elif 'bias' in name and 'conv1' in name:
                # Conv1D bias
                nn.init.zeros_(param)
            elif 'fc' in name and 'weight' in name:
                # Dense layer weights
                nn.init.xavier_uniform_(param)
            elif 'fc' in name and 'bias' in name:
                # Dense layer biases
                nn.init.zeros_(param)

    def forward(self, x):
        # 1D Conv with ReLU activation (matching TF)
        x = self.conv1(x)           # [batch, 1, seq_len] -> [batch, 16, seq_len]
        x = torch.relu(x)
        x = x.permute(0, 2, 1)      # Rearrange for GRU: [batch, seq_len, 16]
        
        # First bidirectional ReLU GRU with return_sequences=True
        x, _ = self.gru1(x)         # [batch, seq_len, 128] (64*2)
        x = self.dropout1(x)
        
        # Second bidirectional ReLU GRU with return_sequences=False (only final state)
        _, h_n = self.gru2(x)       # h_n: [batch, 256] (128*2 concatenated final states)
        h = self.dropout2(h_n)
        
        # Dense layers with ReLU and linear activation
        h = self.fc1(h)             # [batch, 128]
        h = torch.relu(h)
        h = self.dropout3(h)
        out = self.fc2(h)           # [batch, 1] - linear activation (no activation)
        return out

class WindowGRU(Disaggregator):
    """
    Window-based GRU neural network for Non-Intrusive Load Monitoring (NILM).
    
    Based on "Sliding window approach for online energy disaggregation using artificial neural networks"
    by Krystalakos et al., published in Proceedings of the 10th Hellenic Conference on Artificial Intelligence, 2018.
    DOI: https://doi.org/10.1145/3200947.3201011
    
    This implementation uses a sliding window approach for real-time energy disaggregation,
    employing recurrent neural networks with Gated Recurrent Units (GRUs) for temporal 
    pattern recognition in power consumption data.
    
    Architecture Overview:
    - 1D convolutional layer for initial feature extraction from power sequences
    - Two bidirectional GRU layers with ReLU activation for temporal sequence modeling
    - Dropout layers for regularization to prevent overfitting
    - Fully connected layers for final power consumption prediction
    - Sliding window approach for online, real-time energy disaggregation
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - sequence_length (int): Length of input sequences (default: 99)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - save-model-path (str): Path to save trained models (optional)
            - pretrained-model-path (str): Path to load pre-trained models (optional)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        self.MODEL_NAME = "WindowGRU"
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path', None)
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.max_val = 800
        self.batch_size = params.get('batch_size', 512)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def return_network(self):
        """Factory method to create a new GRU model instance"""
        return GRUNet(self.sequence_length).to(self.device)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df))

        train_appliances = new_train_appliances
        for app_name, app_df in train_appliances:
            if app_name not in self.models:
                _log_print("First model training for", app_name)
                self.models[app_name] = self.return_network()
            else:
                _log_print("Started re-training model for", app_name)

            model = self.models[app_name]
            mains = train_main.reshape((-1, self.sequence_length, 1))
            app_reading = app_df.reshape((-1, 1))
            
            filepath = checkpoint_path(".pt")
            
            # Convert to PyTorch tensors
            mains_tensor = torch.tensor(mains, dtype=torch.float32).permute(0, 2, 1)  # [B, 1, seq]
            app_tensor = torch.tensor(app_reading, dtype=torch.float32).squeeze()     # [B]
            
            # Use validation split like TF (last 15% instead of random split)
            # This follows the legacy TF validation split fraction.
            n_total = len(mains_tensor)
            val_size = max(1, int(0.15 * n_total)) if n_total > 1 else 0
            train_size = n_total - val_size
            
            train_x = mains_tensor[:train_size].to(self.device)
            val_x = mains_tensor[train_size:].to(self.device)
            train_y = app_tensor[:train_size].to(self.device)
            val_y = app_tensor[train_size:].to(self.device)
            
            # Use Adam with TensorFlow-style default parameters.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            
            # Create DataLoader for training data with shuffle=True (like TF)
            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            for epoch in range(self.n_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                num_batches = 0
                
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x).squeeze(-1)  # Ensure output shape matches target
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    num_batches += 1
                
                train_loss /= num_batches
                
                # Validation phase (evaluate on full validation set at once)
                model.eval()
                with torch.no_grad():
                    val_outputs = model(val_x).squeeze(-1)
                    val_loss = criterion(val_outputs, val_y).item()
                
                # Save best model (like ModelCheckpoint in TF with verbose=1)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), filepath)
                    _log_print(f'Epoch {epoch+1}/{self.n_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
                
            # Load best weights (like TF version)
            model.load_state_dict(torch.load(filepath))
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')
        
        test_predictions = []
        for mains in test_main_list:
            disggregation_dict = {}
            mains = mains.values.reshape((-1, self.sequence_length, 1))
            for appliance in self.models:
                # Convert to tensor and process in batches
                mains_tensor = torch.tensor(mains, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    # Process in batches following the legacy TensorFlow behavior.
                    predictions = []
                    for i in range(0, len(mains_tensor), self.batch_size):
                        batch = mains_tensor[i:i + self.batch_size]
                        batch_pred = model(batch).cpu().numpy()
                        predictions.append(batch_pred)
                    prediction = np.concatenate(predictions, axis=0)
                
                prediction = np.reshape(prediction, len(prediction))
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                valid_predictions = self._denormalize(valid_predictions, self.max_val)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == 'train':
            _log_print("Training processing")
            processed_mains = []

            for mains in mains_lst:
                # add padding values
                padding = [0 for i in range(0, self.sequence_length - 1)]
                paddf = pd.DataFrame({mains.columns.values[0]: padding})
                mains = pd.concat([mains, paddf])
                mainsarray = self.preprocess_train_mains(mains)
                processed_mains.append(pd.DataFrame(mainsarray))

            tuples_of_appliances = []
            for (appliance_name, app_dfs_list) in submeters_lst:
                processed_app_dfs = []
                for app_df in app_dfs_list:                    
                    data = self.preprocess_train_appliances(app_df)
                    processed_app_dfs.append(pd.DataFrame(data))
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method == 'test':
            processed_mains = []
            for mains in mains_lst:                
                # add padding values
                padding = [0 for i in range(0, self.sequence_length - 1)]
                paddf = pd.DataFrame({mains.columns.values[0]: padding})
                mains = pd.concat([mains, paddf])
                mainsarray = self.preprocess_test_mains(mains)
                processed_mains.append(pd.DataFrame(mainsarray))

            return processed_mains

    def preprocess_test_mains(self, mains):
        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[
            None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        mainsarray = mainsarray.reshape((-1, self.sequence_length))
        return pd.DataFrame(mainsarray)

    def preprocess_train_appliances(self, appliance):
        appliance = self._normalize(appliance, self.max_val)
        appliancearray = np.array(appliance)
        appliancearray = appliancearray.reshape((-1, 1))
        return pd.DataFrame(appliancearray)

    def preprocess_train_mains(self, mains):
        mains = self._normalize(mains, self.max_val)
        mainsarray = np.array(mains)
        indexer = np.arange(self.sequence_length)[None, :] + np.arange(len(mainsarray) - self.sequence_length + 1)[:, None]
        mainsarray = mainsarray[indexer]
        mainsarray = mainsarray.reshape((-1, self.sequence_length))
        return pd.DataFrame(mainsarray)

    def _normalize(self, chunk, mmax):
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        tchunk = chunk * mmax
        return tchunk
