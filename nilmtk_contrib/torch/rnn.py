from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class RNNModel(nn.Module):
    """
    An RNN-based model for NILM, with an architecture designed to mirror the
    original TensorFlow implementation.
    """
    def __init__(self, sequence_length):
        super(RNNModel, self).__init__()
        self.sequence_length = sequence_length
        
        # Layers are defined to match the TensorFlow architecture
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, 
                                stride=1, padding=2) # 'same' padding
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initializes weights to match TensorFlow's default initializations."""
        # Use Xavier uniform for Conv, LSTM, and Linear layers by default
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        # Input shape: (batch, seq_len, 1) -> permute for Conv1D
        x = x.permute(0, 2, 1)
        
        # Feature extraction
        x = self.conv1d(x)
        
        # Permute for LSTM layers
        x = x.permute(0, 2, 1)
        
        # Sequence processing
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # In the original TF model, only the output of the last time step is used.
        x = x[:, -1, :]
        
        # Final prediction layers
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x

class RNN(Disaggregator):
    """
    RNN disaggregator for Non-Intrusive Load Monitoring (NILM).
    
    Based on "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation"
    (https://arxiv.org/abs/1507.06594). This implementation uses a convolutional
    layer followed by bidirectional LSTM layers to learn temporal patterns in
    aggregate power consumption data and predict individual appliance usage.
    
    The model architecture consists of:
    1. 1D Convolutional layer for feature extraction from power sequences
    2. Two bidirectional LSTM layers for learning long-term dependencies
    3. Fully connected layers for final power regression
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - sequence_length (int): Length of input sequences (default: 19)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - appliance_params (dict): Appliance-specific parameters
            - mains_mean (float): Mean normalization for mains power (default: 1800)
            - mains_std (float): Standard deviation for mains power (default: 600)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        """Initializes the disaggregator and its hyperparameters."""
        self.MODEL_NAME = "RNN"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 19)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        self.mains_mean = params.get('mains_mean', 1800)
        self.mains_std = params.get('mains_std', 600)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd for proper windowing.")

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """Trains the model on a chunk of data."""
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        _log_print("...............RNN partial_fit running...............")
        
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        # Prepare data for training
        train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df_values = pd.concat(app_dfs, axis=0).values.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                _log_print(f"First time training for {appliance_name}")
                self.models[appliance_name] = self.return_network()
            else:
                _log_print(f"Retraining model for {appliance_name}")

            model = self.models[appliance_name]
            if train_main.size > 10:
                    filepath = checkpoint_path(".pt")
                    
                    # Convert to PyTorch Tensors
                    train_main_tensor = torch.tensor(train_main, dtype=torch.float32)
                    power_tensor = torch.tensor(power, dtype=torch.float32).squeeze()
                    
                    # Use the last 15% of data for validation to mirror TensorFlow's behavior
                    val_size = max(1, int(0.15 * len(train_main_tensor))) if len(train_main_tensor) > 1 else 0
                    train_size = len(train_main_tensor) - val_size
                    
                    train_x = train_main_tensor[:train_size].to(self.device)
                    val_x = train_main_tensor[train_size:].to(self.device)
                    train_y = power_tensor[:train_size].to(self.device)
                    val_y = power_tensor[train_size:].to(self.device)
                    
                    # Optimizer and loss function, with parameters matching TensorFlow
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)
                    criterion = nn.MSELoss()
                    
                    best_val_loss = float('inf')
                    
                    # Create DataLoader for batching
                    train_dataset = TensorDataset(train_x, train_y)
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    
                    for epoch in range(self.n_epochs):
                        # --- Training Phase ---
                        model.train()
                        train_loss = 0.0
                        
                        for batch_x, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_x).squeeze(-1)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                        
                        train_loss /= len(train_loader)
                        
                        # --- Validation Phase ---
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(val_x).squeeze(-1)
                            val_loss = criterion(val_outputs, val_y).item()
                        
                        # Save the best model based on validation loss
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), filepath)
                            _log_print(f'Epoch {epoch+1}/{self.n_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
                        
                    # Load the best performing model
                    model.load_state_dict(torch.load(filepath))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregates a chunk of mains data."""
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            
            for appliance, model in self.models.items():
                test_tensor = torch.tensor(test_main_array, dtype=torch.float32).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    # Process in batches to manage memory
                    predictions = []
                    for i in range(0, len(test_tensor), self.batch_size):
                        batch = test_tensor[i:i + self.batch_size]
                        batch_pred = model(batch).cpu().numpy()
                        predictions.append(batch_pred)
                    prediction = np.concatenate(predictions, axis=0)
                
                # Denormalize the prediction
                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']
                denormalized_prediction = app_mean + (prediction * app_std)
                
                # Set negative values to zero
                denormalized_prediction[denormalized_prediction < 0] = 0
                df = pd.Series(denormalized_prediction.flatten())
                disggregation_dict[appliance] = df
                
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):
        """Returns a new, initialized RNNModel instance."""
        model = RNNModel(self.sequence_length).to(self.device)
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """
        Preprocesses data by windowing and normalizing, mirroring the
        original TensorFlow implementation.
        """
        if method == 'train':
            # Preprocess mains
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))

            # Preprocess appliances
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                if app_name not in self.appliance_params:
                    raise ApplianceNotFoundError(f"Parameters for appliance '{app_name}' not found!")
                
                app_mean = self.appliance_params[app_name]['mean']
                app_std = self.appliance_params[app_name]['std']

                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_app_dfs))
            return processed_mains_lst, appliance_list

        else: # method == 'test'
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
        for (app_name, df_list) in train_appliances:
            values = np.concatenate([df.values for df in df_list])
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100  # Avoid division by zero for flat signals
            self.appliance_params[app_name] = {'mean': app_mean, 'std': app_std}
        _log_print("Appliance parameters set:", self.appliance_params)