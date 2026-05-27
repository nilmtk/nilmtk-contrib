from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from nilmtk.disaggregate import Disaggregator


from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class SequenceLengthError(Exception):
    pass


class ApplianceNotFoundError(Exception):
    pass


class MSDCNet(nn.Module):
    """
    MSDC Neural Network with a dual-branch CNN architecture.
    This model is based on the S2S_state model from the official MSDC repository.
    
    - Branch 1: Predicts power consumption for each appliance state.
    - Branch 2: Predicts the appliance state.
    """
    
    def __init__(self, window_length, out_len, num_states):
        super(MSDCNet, self).__init__()
        self.window_length = window_length
        self.out_len = out_len
        self.num_states = num_states
        
        # Power branch (Branch 1) - following original MSDC architecture
        self.conv1_p = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_p = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_p = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_p = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_p = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_p = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_p = nn.Linear(60 * window_length, 1024)
        self.fc2_p = nn.Linear(1024, out_len * num_states)
        
        # State branch (Branch 2) - following original MSDC architecture
        self.conv1_s = nn.Conv1d(1, 30, 13, padding=6)
        self.conv2_s = nn.Conv1d(30, 30, 11, padding=5)
        self.conv3_s = nn.Conv1d(30, 40, 7, padding=3)
        self.conv4_s = nn.Conv1d(40, 50, 5, padding=2)
        self.conv5_s = nn.Conv1d(50, 60, 5, padding=2)
        self.conv6_s = nn.Conv1d(60, 60, 5, padding=2)
        self.fc1_s = nn.Linear(60 * window_length, 1024)
        self.fc2_s = nn.Linear(1024, out_len * num_states)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, window_length)
        
        Returns:
            power_preds: Power predictions for each state (batch_size, out_len * num_states)
            state_preds: State classification scores (batch_size, out_len * num_states)
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, window_length)
        y = x
        
        # Power branch
        x = F.relu(self.conv1_p(x))
        x = F.relu(self.conv2_p(x))
        x = F.relu(self.conv3_p(x))
        x = F.relu(self.conv4_p(x))
        x = F.relu(self.conv5_p(x))
        x = F.relu(self.conv6_p(x))
        x = x.flatten(-2, -1)
        x = F.relu(self.fc1_p(x))
        power_preds = self.fc2_p(x)
        
        # State branch
        y = F.relu(self.conv1_s(y))
        y = F.relu(self.conv2_s(y))
        y = F.relu(self.conv3_s(y))
        y = F.relu(self.conv4_s(y))
        y = F.relu(self.conv5_s(y))
        y = F.relu(self.conv6_s(y))
        y = y.flatten(-2, -1)
        y = F.relu(self.fc1_s(y))
        state_preds = self.fc2_s(y)
        
        return power_preds, state_preds


class MSDC(Disaggregator):
    """
    Multi-State Dual CNN for non-intrusive load monitoring without CRF layer.
    
    This implementation is based on the paper:
    "MSDC: Exploiting Multi-State Power Consumption in Non-intrusive Load Monitoring based on A Dual-CNN Model"
    https://arxiv.org/abs/2302.05565
    
    The model uses a dual-branch CNN architecture without the CRF layer for joint state 
    classification and power prediction in energy disaggregation tasks. This version 
    directly predicts states and power consumption without CRF-based transition modeling.
    
    Architecture Overview:
    - Dual-branch CNN for feature extraction
    - Branch 1: Power consumption prediction for each state
    - Branch 2: Direct state classification (without CRF layer)
    - Multi-state power consumption modeling
    - Simplified architecture compared to full MSDC model
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences
            - n_epochs (int): Number of training epochs
            - batch_size (int): Training batch size
            - appliance_params (dict): Appliance-specific normalization parameters
    """
    
    # Complete dataset-specific configurations from official MSDC implementation
    APPLIANCE_STATES = {
        'kettle': {
            'uk_dale': {
                'states': [2000, 4500],
                'state_averages': [1.15, 2280.79],
                'num_states': 2,
                'threshold': 2000
            }
            # No REDD config for kettle in original - will fallback to UK-DALE
        },
        'microwave': {
            'uk_dale': {
                'states': [300, 3000],
                'state_averages': [1.4, 1551.3],
                'num_states': 2,
                'threshold': 300
            },
            'redd': {
                'states': [300, 3000],
                'state_averages': [4.2, 1557.501],
                'num_states': 2,
                'threshold': 300
            }
        },
        'fridge': {
            'uk_dale': {
                'states': [20, 200, 2500],
                'state_averages': [0.13, 87.26, 246.5],
                'num_states': 3,
                'threshold': 20
            },
            'redd': {
                'states': [50, 300, 500],
                'state_averages': [3.2, 143.3, 397.3],
                'num_states': 3,
                'threshold': 50
            },
            'redd_house1': {
                'states': [50, 300, 500],
                'state_averages': [6.49, 192.57, 443],
                'num_states': 3,
                'threshold': 50
            },
            'redd_house2': {
                'states': [50, 300, 500],
                'state_averages': [6.34, 162.87, 418.36],
                'num_states': 3,
                'threshold': 50
            },
            'redd_house3': {
                'states': [50, 300, 500],
                'state_averages': [0.54, 118.85, 409.75],
                'num_states': 3,
                'threshold': 50
            }
        },
        'dishwasher': {
            'uk_dale': {
                'states': [50, 1000, 4500],
                'state_averages': [0.89, 122.56, 2324.9],
                'num_states': 3,
                'threshold': 50
            },
            'redd': {
                'states': [150, 300, 1000, 3000],
                'state_averages': [0.57, 232.91, 733.89, 1198.31],
                'num_states': 4,
                'threshold': 150
            },
            'redd_house1': {
                'states': [150, 300, 1000, 3000],
                'state_averages': [0.21, 216.75, 438.51, 1105.08],
                'num_states': 4,
                'threshold': 150
            },
            'redd_house2': {
                'states': [150, 1000, 3000],
                'state_averages': [0.16, 250.26, 1197.93],
                'num_states': 3,
                'threshold': 150
            },
            'redd_house3': {
                'states': [50, 400, 1000],
                'state_averages': [0.97, 195.6, 743.42],
                'num_states': 3,
                'threshold': 50
            }
        },
        'washing machine': {
            'uk_dale': {
                'states': [50, 800, 3500],
                'state_averages': [0.13, 204.64, 1892.85],
                'num_states': 3,
                'threshold': 50
            },
            'uk_dale_house2': {
                'states': [50, 200, 1000, 4000],
                'state_averages': [2.83, 114.34, 330.25, 2100.14],
                'num_states': 4,
                'threshold': 50
            },
            'redd': {
                'states': [500, 5000],
                'state_averages': [0, 2627.3],
                'num_states': 2,
                'threshold': 500
            }
        }
    }
    
    # Dataset-specific normalization parameters
    DATASET_NORMALIZATION = {
        'uk_dale': {
            'mains_mean': 1800,
            'mains_std': 600
        },
        'redd': {
            'mains_mean': 352.32,  # From official MSDC REDD implementation
            'mains_std': 608.42
        }
    }
    
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        super().__init__()
        
        self.MODEL_NAME = "MSDC"
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Dataset configuration
        self.dataset = params.get('dataset', 'uk_dale').lower()
        self.house = params.get('house', None)
        
        # Validate dataset
        if self.dataset not in ['uk_dale', 'redd']:
            _log_print(f"Warning: Unknown dataset '{self.dataset}'. Defaulting to 'uk_dale'.")
            self.dataset = 'uk_dale'
        
        # Build dataset key for configuration lookup
        if self.house is not None:
            self.dataset_key = f"{self.dataset}_house{self.house}"
        else:
            self.dataset_key = self.dataset
        
        # Extract hyperparameters
        self.sequence_length = params.get('sequence_length', 99)
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd")
            
        # Output length for sequence-to-sequence prediction
        self.out_len = params.get('out_len', 64)
        self.num_states = params.get('num_states', 3)  # Will be overridden by appliance config
        self.n_epochs = params.get('n_epochs', 50)
        self.batch_size = params.get('batch_size', 256)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.patience = params.get('patience', 5)
        
        # Dataset-specific normalization parameters
        dataset_norm = self.DATASET_NORMALIZATION.get(self.dataset, self.DATASET_NORMALIZATION['uk_dale'])
        self.mains_mean = params.get('mains_mean', dataset_norm['mains_mean'])
        self.mains_std = params.get('mains_std', dataset_norm['mains_std'])
        self.appliance_params = params.get('appliance_params', {})
        
        # Model storage
        self.models = OrderedDict()  # Store separate models for each appliance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Display configuration
        _log_print(f"MSDC initialized for dataset: {self.dataset.upper()}")
        if self.house:
            _log_print(f"House: {self.house}")
        _log_print(f"Configuration key: {self.dataset_key}")
        _log_print(f"Mains normalization - mean: {self.mains_mean}, std: {self.mains_std}")
    
    def _get_appliance_config(self, appliance_name):
        """Get the best available configuration for an appliance"""
        if appliance_name not in self.APPLIANCE_STATES:
            return None
        
        appliance_configs = self.APPLIANCE_STATES[appliance_name]
        
        # Priority order: dataset_key -> dataset -> any available
        if self.dataset_key in appliance_configs:
            return appliance_configs[self.dataset_key]
        elif self.dataset in appliance_configs:
            return appliance_configs[self.dataset]
        else:
            # Use any available configuration as fallback
            available_configs = list(appliance_configs.keys())
            if available_configs:
                fallback_key = available_configs[0]
                _log_print(f"Warning: No {self.dataset_key} config for {appliance_name}, using {fallback_key}")
                return appliance_configs[fallback_key]
        
        return None
    
    def return_network(self, appliance_name):
        """Factory method to create a new MSDC model instance for specific appliance"""
        config = self._get_appliance_config(appliance_name)
        if config:
            num_states = config['num_states']
            _log_print(f"Creating network for {appliance_name} with {num_states} states ({self.dataset_key})")
        else:
            num_states = self.num_states  # fallback to default
            _log_print(f"Warning: No config found for {appliance_name}, using default {num_states} states")
        
        return MSDCNet(self.sequence_length, self.out_len, num_states).to(self.device)
    
    def set_appliance_params(self, train_appliances):
        """Compute normalization statistics for each appliance from training data"""
        for name, lst in train_appliances:
            # Always compute normalization from training data
            arr = pd.concat(lst, axis=0).values.flatten()
            m, s = arr.mean(), arr.std()
            # Prevent division by zero
            if s < 1:
                s = 100
            _log_print(f"Computed normalization for {name}: mean={m:.2f}, std={s:.2f}")
            
            self.appliance_params[name] = {'mean': m, 'std': s}
    
    def _create_state_labels(self, power_sequence, appliance_name):
        """
        Create state labels using the dataset-specific state dictionary
        """
        power = power_sequence.flatten()
        
        # Get appliance configuration
        config = self._get_appliance_config(appliance_name)
        
        if config:
            thresholds = config['states']
            num_states = config['num_states']
        else:
            # Fallback to dynamic thresholds
            if appliance_name in self.appliance_params:
                params = self.appliance_params[appliance_name]
                mean_power = params['mean']
            else:
                mean_power = power.mean()
            
            num_states = self.num_states
            
            if num_states == 2:
                thresholds = [0.1 * mean_power]
            elif num_states == 3:
                thresholds = [0.1 * mean_power, 0.7 * mean_power]
            else:
                thresholds = np.linspace(0, mean_power * 1.2, num_states)[1:]
        
        # Create state labels based on thresholds
        states = np.zeros_like(power, dtype=np.int64)
        
        for i, threshold in enumerate(thresholds):
            states[power >= threshold] = i + 1
        
        # Ensure states are within valid range
        states = np.clip(states, 0, num_states - 1)
        
        return states.astype(np.int64)
    
    def _compute_msdc_loss(self, power_preds, state_preds, y_power, y_states, appliance_name):
        """
        Computes the combined loss for the MSDC model.
        The loss is a sum of:
        1. Mean Squared Error (MSE) for the final power prediction.
        2. Cross-entropy loss for the state classification.
        """
        batch_size = y_power.shape[0]
        
        # Get number of states for this appliance
        config = self._get_appliance_config(appliance_name)
        if config:
            num_states = config['num_states']
        else:
            num_states = self.num_states
        
        # Reshape predictions: (batch_size, out_len, num_states)
        power_preds = power_preds.view(batch_size, self.out_len, num_states)
        state_preds = state_preds.view(batch_size, self.out_len, num_states)
        
        # Apply softmax to state predictions to get probabilities
        state_probs = F.softmax(state_preds, dim=-1)
        
        # Final power prediction: weighted sum over states
        final_power = torch.sum(state_probs * power_preds, dim=-1, keepdim=False)
        
        # 1. Final power MSE loss
        power_loss = F.mse_loss(final_power, y_power)
        
        # 2. State classification loss
        # Flatten for cross-entropy: (batch_size * out_len, num_states)
        state_preds_flat = state_preds.view(-1, num_states)
        y_states_flat = y_states.view(-1)
        state_loss = F.cross_entropy(state_preds_flat, y_states_flat)
        
        # Combined loss (following original implementation)
        total_loss = power_loss + state_loss
        
        return total_loss, power_loss, state_loss

    def partial_fit(self, train_main, train_appliances, 
                    do_preprocessing=True, current_epoch=0, **_):
        """Train MSDC models on a chunk of data"""

        _log_print("Started Partial Fit")
        
        # Compute appliance parameters if not provided
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        
        _log_print("Preprocessing called")
        # Preprocess data using NILMTK-compatible method
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
            
        _log_print("Preprocessing done")
        
        # Prepare main power data
        mains_arr = pd.concat(train_main, axis=0).values
        if len(mains_arr.shape) == 2:
            mains_arr = mains_arr.reshape(-1, self.sequence_length)
        else:
            mains_arr = mains_arr.reshape(-1, self.sequence_length)
        
        # Prepare appliance data
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values
            if len(app_df_values.shape) == 2:
                app_df_values = app_df_values.reshape(-1, self.out_len)
            else:
                app_df_values = app_df_values.reshape(-1, self.out_len)
            new_train_appliances.append((app_name, app_df_values))
        
        train_appliances = new_train_appliances
        
        # Train a separate model for each appliance
        for appliance_name, app_data in train_appliances:
            _log_print(f"\nTraining {appliance_name} for {self.dataset_key}...")
            
            # Check if the appliance was already trained
            if appliance_name not in self.models:
                self.models[appliance_name] = self.return_network(appliance_name)
            
            model = self.models[appliance_name]
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Convert to tensors
            mains_tensor = torch.FloatTensor(mains_arr).to(self.device)
            app_tensor = torch.FloatTensor(app_data).to(self.device)
            
            # Create state labels for each sequence using dataset-specific states
            state_labels = []
            for i in range(app_data.shape[0]):
                states = self._create_state_labels(app_data[i], appliance_name)
                state_labels.append(states)
            state_labels = np.array(state_labels)
            state_tensor = torch.LongTensor(state_labels).to(self.device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(mains_tensor, app_tensor, state_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training loop
            model.train()
            _log_print("Training loop started")
            for epoch in range(self.n_epochs):
                _log_print(f"Epoch {epoch + 1}/{self.n_epochs} for {appliance_name}")
                total_loss = 0
                batch_count = 0
                for batch_mains, batch_app, batch_states in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass through MSDC network
                    power_preds, state_preds = model(batch_mains)
                    
                    # Compute MSDC loss (without CRF)
                    loss, power_loss, state_loss = self._compute_msdc_loss(
                        power_preds, state_preds, batch_app, batch_states, appliance_name
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if epoch % 10 == 0:
                    avg_loss = total_loss / batch_count
                    _log_print(f"Epoch {epoch}/{self.n_epochs}, Avg Loss: {avg_loss:.4f}")
    
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregate power consumption using the trained MSDC model."""
        
        if model is not None:
            self.models = model
        
        # Preprocess the test mains
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')
        
        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length))
            disggregation_dict = {}
            
            test_main_tensor = torch.FloatTensor(test_main).to(self.device)
            
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                
                # Get appliance configuration
                config = self._get_appliance_config(appliance)
                if config:
                    num_states = config['num_states']
                else:
                    num_states = self.num_states
                
                with torch.no_grad():
                    # Forward pass through MSDC
                    power_preds, state_preds = model(test_main_tensor)
                    
                    # Reshape predictions
                    batch_size = power_preds.shape[0]
                    power_preds = power_preds.view(batch_size, self.out_len, num_states)
                    state_preds = state_preds.view(batch_size, self.out_len, num_states)
                    
                    # Apply softmax to get state probabilities
                    state_probs = F.softmax(state_preds, dim=-1)
                    
                    # Final power prediction: weighted sum over states
                    predicted_power = torch.sum(state_probs * power_preds, dim=-1)
                    
                    # Extract center values (middle of each window)
                    center_idx = self.out_len // 2
                    pred = predicted_power[:, center_idx].cpu().numpy()
                    
                    # Denormalize predictions
                    pred = pred * self.appliance_params[appliance]['std'] + self.appliance_params[appliance]['mean']
                    pred = np.where(pred > 0, pred, 0)  # Ensure non-negative power
                
                disggregation_dict[appliance] = pred
            
            test_predictions.append(pd.DataFrame(disggregation_dict, dtype='float32'))
        
        return test_predictions
    
    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """
        Preprocessing method required by NILMTK API
        """
        if method == 'train':
            # Process mains data
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            
            # Process appliance data - create sequence-to-sequence targets
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    raise ApplianceNotFoundError()
                
                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    n = self.sequence_length
                    units_to_pad = n // 2
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                    
                    # Create sequence-to-sequence targets (out_len length)
                    app_sequences = []
                    offset = int(0.5 * (self.sequence_length - 1.0))
                    for i in range(len(new_app_readings) - self.sequence_length + 1):
                        # Extract output sequence from center
                        start_idx = i + offset - self.out_len // 2
                        end_idx = start_idx + self.out_len
                        if start_idx >= 0 and end_idx <= len(new_app_readings):
                            seq = new_app_readings[start_idx:end_idx]
                        else:
                            # Pad if necessary
                            seq = np.zeros(self.out_len)
                            if start_idx < 0:
                                seq[-start_idx:] = new_app_readings[0:end_idx]
                            elif end_idx > len(new_app_readings):
                                seq[:len(new_app_readings)-start_idx] = new_app_readings[start_idx:]
                            else:
                                seq = new_app_readings[start_idx:end_idx]
                        
                        app_sequences.append(seq)
                    
                    app_sequences = np.array(app_sequences)
                    app_sequences = (app_sequences - app_mean) / app_std
                    processed_app_dfs.append(pd.DataFrame(app_sequences))
                
                appliance_list.append((app_name, processed_app_dfs))
            
            return processed_mains_lst, appliance_list
        
        else:  # method == 'test'
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

# Export for nilmtk_contrib
__all__ = ['MSDC']
