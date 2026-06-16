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
    Dual-branch CNN for joint state classification and power prediction.
    - Branch 1: Predicts state emission scores for a CRF.
    - Branch 2: Predicts power consumption for each state.
    - CRF layer models state transitions.
    """
    
    def __init__(self, window_length, num_states):
        super(MSDCNet, self).__init__()
        self.window_length = window_length
        self.num_states = num_states
        
        # Shared CNN feature extractor
        self.shared_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 1: State emission scores for CRF
        self.state_branch = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_states)
        )
        
        # Branch 2: Power predictions for each state
        self.power_branch = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_states)
        )
        
        # CRF layer for state sequence modeling
        self.crf = CRF(num_states)
    
    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, seq_len, window_length)
        
        Returns:
            emissions: State emission scores (batch_size, seq_len, num_states)
            power_preds: Power predictions for each state (batch_size, seq_len, num_states)
        """
        batch_size, seq_len, window_length = x.shape
        
        # Reshape for CNN: (batch_size * seq_len, 1, window_length)
        x_reshaped = x.view(-1, 1, window_length)
        
        # Extract features using shared CNN
        features = self.shared_cnn(x_reshaped)  # (batch_size * seq_len, 64, 1)
        features = features.squeeze(-1)  # (batch_size * seq_len, 64)
        
        # Branch 1: State emissions
        emissions = self.state_branch(features)  # (batch_size * seq_len, num_states)
        emissions = emissions.view(batch_size, seq_len, self.num_states)
        
        # Branch 2: Power predictions
        power_preds = self.power_branch(features)  # (batch_size * seq_len, num_states)
        power_preds = power_preds.view(batch_size, seq_len, self.num_states)
        
        return emissions, power_preds


class CRF(nn.Module):
    """Conditional Random Field for sequence modeling."""
    
    def __init__(self, num_states):
        super(CRF, self).__init__()
        self.num_states = num_states
        
        # Transition parameters
        self.transitions = nn.Parameter(torch.randn(num_states, num_states))
        self.start_transitions = nn.Parameter(torch.randn(num_states))
        self.end_transitions = nn.Parameter(torch.randn(num_states))
    
    def forward(self, emissions):
        """Computes the log partition function using the forward algorithm."""
        batch_size, seq_len, num_states = emissions.shape
        
        # Initialize with start transitions
        alpha = emissions[:, 0] + self.start_transitions.unsqueeze(0)
        
        # Forward pass
        for t in range(1, seq_len):
            alpha_expanded = alpha.unsqueeze(2)  # (batch_size, num_states, 1)
            trans_scores = alpha_expanded + self.transitions.unsqueeze(0)  # (batch_size, num_states, num_states)
            alpha = torch.logsumexp(trans_scores, dim=1) + emissions[:, t]
        
        # Add end transitions
        log_partition = torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)
        return log_partition
    
    def score_sequence(self, emissions, states):
        """Computes the log-likelihood of a given state sequence."""
        batch_size, seq_len = states.shape
        
        # Start transition score
        score = self.start_transitions[states[:, 0]]
        
        # Emission scores
        for t in range(seq_len):
            score += emissions[range(batch_size), t, states[:, t]]
        
        # Transition scores
        for t in range(seq_len - 1):
            score += self.transitions[states[:, t], states[:, t + 1]]
        
        # End transition score
        score += self.end_transitions[states[:, -1]]
        
        return score
    
    def viterbi_decode(self, emissions):
        """Finds the most likely state sequence using the Viterbi algorithm."""
        batch_size, seq_len, num_states = emissions.shape
        
        # Initialize
        delta = emissions[:, 0] + self.start_transitions.unsqueeze(0)
        psi = torch.zeros(batch_size, seq_len, num_states, dtype=torch.long, device=emissions.device)
        
        # Forward pass
        for t in range(1, seq_len):
            delta_expanded = delta.unsqueeze(2)  # (batch_size, num_states, 1)
            trans_scores = delta_expanded + self.transitions.unsqueeze(0)  # (batch_size, num_states, num_states)
            
            delta_next, psi[:, t] = torch.max(trans_scores, dim=1)
            delta = delta_next + emissions[:, t]
        
        # Add end transitions and find best final state
        final_scores = delta + self.end_transitions.unsqueeze(0)
        best_final_states = torch.argmax(final_scores, dim=1)
        
        # Backward pass to reconstruct path
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=emissions.device)
        best_paths[:, -1] = best_final_states
        
        for t in range(seq_len - 2, -1, -1):
            best_paths[:, t] = psi[range(batch_size), t + 1, best_paths[:, t + 1]]
        
        return best_paths


class MSDC(Disaggregator):
    """
    Multi-State Dual CNN for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "MSDC: Exploiting Multi-State Power Consumption in Non-intrusive Load Monitoring based on A Dual-CNN Model"
    https://arxiv.org/abs/2302.05565
    
    The model uses a dual-branch CNN architecture with a CRF layer for joint state 
    classification and power prediction in energy disaggregation tasks.
    
    Architecture Overview:
    - Dual-branch CNN for feature extraction
    - Branch 1: State emission scores for CRF layer
    - Branch 2: Power consumption prediction for each state
    - CRF layer for modeling state transitions
    - Multi-state power consumption modeling
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences
            - n_epochs (int): Number of training epochs
            - batch_size (int): Training batch size
            - appliance_params (dict): Appliance-specific normalization parameters
    """
    
    # Dataset-specific configurations from the official MSDC implementation
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
        'washingmachine': {
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
        
        # Validate and build dataset key
        if self.dataset not in ['uk_dale', 'redd']:
            _log_print(f"Warning: Unknown dataset '{self.dataset}'. Defaulting to 'uk_dale'.")
            self.dataset = 'uk_dale'
        
        self.dataset_key = f"{self.dataset}_house{self.house}" if self.house else self.dataset
        
        # Hyperparameters
        self.sequence_length = params.get('sequence_length', 99)
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd")
            
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
        
        # Model and device configuration
        self.models = OrderedDict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Display configuration
        _log_print(f"MSDC initialized for dataset: {self.dataset.upper()}")
        if self.house:
            _log_print(f"House: {self.house}")
        _log_print(f"Configuration key: {self.dataset_key}")
        _log_print(f"Mains normalization - mean: {self.mains_mean}, std: {self.mains_std}")
    
    def _get_appliance_config(self, appliance_name):
        """Retrieves the best available configuration for an appliance."""
        if appliance_name not in self.APPLIANCE_STATES:
            return None
        
        appliance_configs = self.APPLIANCE_STATES[appliance_name]
        
        # Priority: specific house > dataset > any available config
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
        """Creates an MSDC model instance for a specific appliance."""
        config = self._get_appliance_config(appliance_name)
        if config:
            num_states = config['num_states']
            _log_print(f"Creating network for {appliance_name} with {num_states} states ({self.dataset_key})")
        else:
            num_states = self.num_states  # fallback to default
            _log_print(f"Warning: No config found for {appliance_name}, using default {num_states} states")
        
        return MSDCNet(self.sequence_length, num_states).to(self.device)
    
    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
        for name, lst in train_appliances:
            arr = pd.concat(lst, axis=0).values.flatten()
            m, s = arr.mean(), arr.std()
            # Avoid division by zero
            if s < 1:
                s = 100
            _log_print(f"Computed normalization for {name}: mean={m:.2f}, std={s:.2f}")
            
            self.appliance_params[name] = {'mean': m, 'std': s}
    
    def _create_state_labels(self, power_sequence, appliance_name):
        """
        Generates state labels based on dataset-specific configurations.
        """
        power = power_sequence.flatten()
        
        # Get appliance configuration
        config = self._get_appliance_config(appliance_name)
        
        if config:
            thresholds = config['states']
            num_states = config['num_states']
        else:
            # Fallback to dynamic thresholds if no config is found
            mean_power = self.appliance_params.get(appliance_name, {}).get('mean', power.mean())
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
    
    def _compute_msdc_loss(self, model, x, y_power, y_states, appliance_name):
        """
        Computes the combined MSDC loss.
        - CRF negative log-likelihood for state sequence.
        - MSE for per-state power predictions.
        - MSE for final power prediction based on Viterbi-decoded states.
        """
        # Forward pass
        emissions, power_preds = model(x)
        
        # Use the model's CRF
        crf = model.crf
        
        # Get number of states for the appliance
        config = self._get_appliance_config(appliance_name)
        num_states = config['num_states'] if config else self.num_states
        
        # 1. CRF loss (negative log-likelihood)
        log_partition = crf(emissions)
        sequence_scores = crf.score_sequence(emissions, y_states)
        crf_loss = torch.mean(log_partition - sequence_scores)
        
        # 2. Per-state power loss
        batch_size, seq_len = y_states.shape
        state_power_loss = 0
        for state_id in range(num_states):
            state_mask = (y_states == state_id).float()
            if state_mask.sum() > 0:
                state_power_pred = power_preds[:, :, state_id]
                masked_pred = state_power_pred * state_mask
                masked_target = y_power * state_mask
                state_power_loss += F.mse_loss(masked_pred, masked_target, reduction='sum') / (state_mask.sum() + 1e-8)
        
        # 3. Final power loss (using Viterbi-decoded states)
        best_states = crf.viterbi_decode(emissions)
        final_power_pred = torch.zeros_like(y_power)
        for b in range(batch_size):
            for t in range(seq_len):
                state = best_states[b, t]
                final_power_pred[b, t] = power_preds[b, t, state]
        
        final_power_loss = F.mse_loss(final_power_pred, y_power)
        
        # Combined loss with weights from the paper
        total_loss = crf_loss + 0.5 * state_power_loss + final_power_loss
        
        return total_loss, crf_loss, state_power_loss, final_power_loss

    def partial_fit(self, train_main, train_appliances, 
                    do_preprocessing=True, current_epoch=0, **_):
        """Trains the model on a chunk of data."""

        _log_print("started Partial Fit")
        
        # Set appliance parameters if not already done
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        
        # Preprocess data
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
            
        _log_print("Preprocessing done")
        
        # Prepare main power data
        mains_arr = pd.concat(train_main, axis=0).values
        if len(mains_arr.shape) == 2:
            mains_arr = mains_arr.reshape(-1, self.sequence_length, 1)
        else:
            mains_arr = mains_arr.reshape(-1, self.sequence_length, 1)
        
        # Prepare appliance data
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values
            new_train_appliances.append((app_name, app_df_values))
        
        train_appliances = new_train_appliances
        
        # Train a separate model for each appliance
        for appliance_name, app_data in train_appliances:
            _log_print(f"\nTraining MSDC for {appliance_name}...")
            
            # Initialize model if not already trained
            if appliance_name not in self.models:
                self.models[appliance_name] = self.return_network(appliance_name)
            
            model = self.models[appliance_name]
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # Convert data to tensors
            mains_tensor = torch.FloatTensor(mains_arr).to(self.device)
            app_tensor = torch.FloatTensor(app_data).to(self.device)
            
            # Create state labels
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
            _log_print(f"Training on {self.device}...")
            for epoch in range(self.n_epochs):
                _log_print(f"Epoch {epoch + 1}/{self.n_epochs} for {appliance_name}")
                total_loss = 0
                batch_count = 0
                for batch_mains, batch_app, batch_states in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    emissions, power_preds = model(batch_mains)
                    
                    # Compute loss
                    loss, crf_loss, state_power_loss, final_power_loss = self._compute_msdc_loss(
                        model, batch_mains, batch_app.squeeze(-1), batch_states, appliance_name
                    )
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if epoch % 10 == 0:
                    avg_loss = total_loss / batch_count
                    _log_print(f"Epoch {epoch}/{self.n_epochs}, Avg Loss: {avg_loss:.4f}")
            
            _log_print(f"Training completed for {appliance_name}!")
    
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregates a chunk of mains data using the trained models."""
        
        if model is not None:
            self.models = model
        
        # Preprocess test data
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')
        
        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            
            test_main_tensor = torch.FloatTensor(test_main).to(self.device)
            
            for appliance, model in self.models.items():
                _log_print(f"Predicting {appliance}...")
                model.eval()
                
                with torch.no_grad():
                    # Forward pass
                    emissions, power_preds = model(test_main_tensor)
                    
                    # Decode state sequence using Viterbi
                    best_states = model.crf.viterbi_decode(emissions)
                    
                    # Get power predictions for the decoded state sequence
                    batch_size, seq_len = best_states.shape
                    predicted_power = torch.zeros(batch_size, seq_len, device=self.device)
                    
                    for b in range(batch_size):
                        for t in range(seq_len):
                            state = best_states[b, t]
                            predicted_power[b, t] = power_preds[b, t, state]
                    
                    # Extract center values (middle of each window)
                    center_idx = self.sequence_length // 2
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
            
            # Process appliance data
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
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                
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