import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class Seq2SeqModel(nn.Module):
    """
    A Sequence-to-Sequence (Seq2Seq) CNN model for NILM, with an architecture
    designed to mirror the original TensorFlow implementation.
    """
    def __init__(self, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        
        # --- Encoder Layers ---
        self.conv1 = nn.Conv1d(1, 30, kernel_size=10, stride=2, padding=0)
        self.conv2 = nn.Conv1d(30, 30, kernel_size=8, stride=2, padding=0)
        self.conv3 = nn.Conv1d(30, 40, kernel_size=6, stride=1, padding=0)
        self.conv4 = nn.Conv1d(40, 50, kernel_size=5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.2)
        self.conv5 = nn.Conv1d(50, 50, kernel_size=5, stride=1, padding=0)
        self.dropout2 = nn.Dropout(0.2)

        # Calculate the flattened size dynamically after convolutions
        self._calculate_flatten_size(sequence_length)

        # --- Decoder Layers ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flat_size, 1024)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, sequence_length)
        
        self._init_weights()

    def _calculate_flatten_size(self, seq_len):
        """Calculates the input size for the decoder's fully connected layer."""
        # Simulate the sequence length reduction through the encoder
        L = seq_len
        L = (L - 10) // 2 + 1
        L = (L - 8) // 2 + 1
        L = L - 6 + 1
        L = L - 5 + 1
        L = L - 5 + 1
        self.flat_size = 50 * L
    
    def _init_weights(self):
        """Initializes weights to match TensorFlow's default (glorot_uniform)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input shape: (batch, seq_len, 1) -> permute for Conv1D
        x = x.permute(0, 2, 1)
        
        # --- Encoder ---
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv5(x))
        x = self.dropout2(x)
        
        # --- Decoder ---
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x) # Linear activation
        return x

class Seq2Seq(Disaggregator):
    """
    Sequence-to-Sequence CNN for Non-Intrusive Load Monitoring (NILM).
    
    Based on the foundational sequence-to-sequence learning approach from:
    "Sequence to Sequence Learning with Neural Networks" by Sutskever et al.
    https://arxiv.org/abs/1409.3215
    
    This implementation adapts the sequence-to-sequence paradigm for energy disaggregation,
    using a CNN-based encoder-decoder architecture instead of the original LSTM approach.
    The model learns to map input sequences of aggregate power consumption to output 
    sequences of individual appliance power consumption.
    
    Architecture Overview:
    - Encoder: Multiple 1D convolutional layers with decreasing stride for feature extraction
    - Decoder: Fully connected layers that reconstruct the sequence from encoded features
    - Dropout layers for regularization throughout the network
    - Sequence-to-sequence learning for temporal power disaggregation
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - sequence_length (int): Length of input/output sequences (default: 99, must be odd)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - appliance_params (dict): Appliance-specific normalization parameters
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        """Initializes the disaggregator and its hyperparameters."""
        self.MODEL_NAME = "Seq2Seq"
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd!")

    def return_network(self):
        """Returns a new, initialized Seq2SeqModel instance."""
        return Seq2SeqModel(self.sequence_length).to(self.device)

    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
        for (app_name, df_list) in train_appliances:
            values = np.concatenate([df.values for df in df_list])
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100 # Avoid division by zero for flat signals
            self.appliance_params[app_name] = {'mean': app_mean, 'std': app_std}

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """Trains the model on a chunk of data."""
        _log_print("...............Seq2Seq partial_fit running...............")
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        # Prepare data for training
        train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df_values = pd.concat(app_dfs, axis=0).values.reshape((-1, self.sequence_length))
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
                    power_tensor = torch.tensor(power, dtype=torch.float32)
                    
                    # Use the last 15% of data for validation to mirror TensorFlow's behavior
                    n_total = len(train_main_tensor)
                    val_size = max(1, int(0.15 * n_total)) if n_total > 1 else 0
                    
                    train_x = train_main_tensor[:-val_size].to(self.device)
                    val_x = train_main_tensor[-val_size:].to(self.device)
                    train_y = power_tensor[:-val_size].to(self.device)
                    val_y = power_tensor[-val_size:].to(self.device)
                    
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
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                        
                        train_loss /= len(train_loader)
                        
                        # --- Validation Phase ---
                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(val_x)
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
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))

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

                # Average predictions over overlapping windows
                window_length = self.sequence_length
                n = len(prediction) + window_length - 1
                sum_arr = np.zeros(n)
                counts_arr = np.zeros(n)
                
                for i, p in enumerate(prediction):
                    sum_arr[i:i+window_length] += p.flatten()
                    counts_arr[i:i+window_length] += 1
                
                # Avoid division by zero
                counts_arr[counts_arr == 0] = 1
                averaged_prediction = sum_arr / counts_arr

                # Denormalize the prediction
                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']
                denormalized_prediction = app_mean + (averaged_prediction * app_std)
                
                # Set negative values to zero
                denormalized_prediction[denormalized_prediction < 0] = 0
                df = pd.Series(denormalized_prediction)
                disggregation_dict[appliance] = df
                
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions

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
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])                    
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                    
                appliance_list.append((app_name, processed_app_dfs))

            return processed_mains_lst, appliance_list

        else: # method == 'test'
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                # The original TF implementation did not pad test data, so we omit it here.
                # units_to_pad = n // 2
                # new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst