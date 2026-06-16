from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) implementation.
    This network uses a series of temporal blocks with dilated, causal convolutions 
    to capture long-range dependencies in sequential data.
    """
    def __init__(self, sequence_length, num_levels=8, num_filters=25, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        
        self.num_levels = num_levels
        self.num_filters = num_filters
        
        layers = []
        num_channels = [1] + [num_filters] * num_levels
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            
            layers.append(TemporalBlock(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                dilation=dilation_size, 
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Final fully connected layer
        self.final_length = self._calculate_output_length(sequence_length, kernel_size, num_levels)
        self.fc = nn.Linear(num_filters * self.final_length, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_output_length(self, input_length, kernel_size, num_levels):
        """Calculates the output length after all temporal blocks."""
        # Causal convolutions with proper padding maintain the sequence length.
        return input_length
    
    def _initialize_weights(self):
        """Initializes weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length) 
        x = self.network(x)
        # Output shape: (batch_size, num_filters, final_length)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class TemporalBlock(nn.Module):
    """
    A single block of a TCN, consisting of two dilated causal convolutions
    with a residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated causal convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        
        # Chomp1d removes padding to ensure causality.
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated causal convolution  
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (with downsampling if channels differ)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
        # Weight normalization for stability
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        if self.downsample is not None:
            self.downsample = nn.utils.weight_norm(self.downsample)
        
        self.init_weights()
    
    def init_weights(self):
        """Initializes weights for the temporal block."""
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
    
    def forward(self, x):
        # First convolution path
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution path
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Add residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure residual and output have the same length
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]
        
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """
    Removes padding from the end of a sequence to make convolutions causal.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TCN(Disaggregator):
    """
    Temporal Convolutional Network (TCN) for Non-Intrusive Load Monitoring (NILM).
    
    Based on "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
    by Bai et al., published in arXiv preprint arXiv:1803.01271, 2018.
    https://arxiv.org/abs/1803.01271
    
    This implementation applies the TCN architecture to energy disaggregation, using dilated causal 
    convolutions to capture long-range temporal dependencies in power consumption sequences. TCNs 
    have been shown to outperform canonical recurrent networks like LSTMs across diverse sequence 
    modeling tasks while demonstrating longer effective memory.
    
    Architecture Overview:
    - Multiple temporal blocks with dilated causal convolutions for long-range dependencies
    - Residual connections within each temporal block for improved gradient flow
    - Dropout layers for regularization to prevent overfitting
    - Sequence-to-point learning for appliance power prediction
    - Exponentially increasing dilation factors to capture patterns at multiple time scales
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - sequence_length (int): Length of input sequences (default: 99, must be odd)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - num_levels (int): Number of temporal blocks (default: 8)
            - num_filters (int): Number of filters per temporal block (default: 25)
            - kernel_size (int): Kernel size for convolutions (default: 7)
            - dropout (float): Dropout rate for regularization (default: 0.2)
            - appliance_params (dict): Appliance-specific normalization parameters
            - mains_mean (float): Mean normalization for mains power (default: 1800)
            - mains_std (float): Standard deviation for mains power (default: 600)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        super().__init__()
        self.MODEL_NAME = "TCN"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Hyperparameters
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        self.mains_mean = params.get("mains_mean", 1800)
        self.mains_std = params.get("mains_std", 600)
        
        # TCN-specific parameters
        self.num_levels = params.get("num_levels", 8)
        self.num_filters = params.get("num_filters", 25)
        self.kernel_size = params.get("kernel_size", 7)
        self.dropout = params.get("dropout", 0.2)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sequence length must be odd for centered windowing.
        if self.sequence_length % 2 == 0:
            _log_print("Sequence length should be odd!")
            raise SequenceLengthError

        _log_print(f"TCN initialized with sequence_length={self.sequence_length}")
        _log_print(f"TCN params: levels={self.num_levels}, filters={self.num_filters}, kernel_size={self.kernel_size}")
        _log_print(f"Using device: {self.device}")

    def return_network(self):
        """Builds and returns the TCN network."""
        model = TemporalConvNet(
            sequence_length=self.sequence_length,
            num_levels=self.num_levels,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        _log_print(f"TCN model created with {total_params:,} parameters")
        
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """Preprocesses data using a sliding window approach."""
        if method == 'train':
            # Preprocess training data
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    raise ApplianceNotFoundError(f"Parameters for appliance '{app_name}' not found!")

                processed_appliance_dfs = []
                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list
        
        else: # method == 'test'
            # Preprocess test data
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
        for app_name, df_list in train_appliances:
            values = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
        _log_print("Appliance parameters set:", self.appliance_params)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """Trains the model on a chunk of data."""
        # Compute appliance parameters if not already set
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        _log_print("...............TCN partial_fit running...............")
        # Preprocess data
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            # Create a new model for the appliance if it's the first time training
            if appliance_name not in self.models:
                _log_print("First time training for", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                _log_print("Retraining model for", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0 and len(train_main) > 10:
                    # Convert to tensors
                    # Conv1d expects (batch, channels, length)
                    train_main_tensor = torch.tensor(train_main, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                    power_tensor = torch.tensor(power, dtype=torch.float32).squeeze().to(self.device)
                    
                    # Create validation split (15%)
                    n_samples = train_main_tensor.size(0)
                    val_size = max(1, int(0.15 * n_samples)) if n_samples > 1 else 0
                    indices = torch.randperm(n_samples)
                    train_idx, val_idx = indices[val_size:], indices[:val_size]
                    
                    train_X = train_main_tensor[train_idx]
                    train_y = power_tensor[train_idx]
                    val_X = train_main_tensor[val_idx]
                    val_y = power_tensor[val_idx]
                    
                    # Setup optimizer and loss function
                    optimizer = torch.optim.Adam(model.parameters())
                    criterion = nn.MSELoss()
                    
                    best_val_loss = float('inf')
                    filepath = checkpoint_path(".pth")
                    
                    # Training loop
                    for epoch in range(self.n_epochs):
                        model.train()
                        
                        # Create data loader for batching
                        train_dataset = TensorDataset(train_X, train_y)
                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                        
                        epoch_losses = []
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            predictions = model(batch_X).squeeze()
                            loss = criterion(predictions, batch_y)
                            loss.backward()
                            
                            # Gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            epoch_losses.append(loss.item())
                        
                        # Validation at the end of each epoch
                        model.eval()
                        with torch.no_grad():
                            val_predictions = model(val_X).squeeze()
                            val_loss = criterion(val_predictions, val_y).item()
                        
                        avg_train_loss = np.mean(epoch_losses)
                        _log_print(f"Epoch {epoch+1}/{self.n_epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
                        
                        # Save the best model based on validation loss
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), filepath)
                            _log_print(f"Validation loss improved, saving model to {filepath}")
                    
                    # Load the best weights after training
                    model.load_state_dict(torch.load(filepath, map_location=self.device))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregates a chunk of mains data."""
        if model is not None:
            self.models = model

        # Preprocess test data
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            
            # Convert to tensor for Conv1d
            test_main_tensor = torch.tensor(test_main, dtype=torch.float32).permute(0, 2, 1).to(self.device)
            
            disggregation_dict = {}
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_tensor).cpu().numpy()
                    # Denormalize predictions
                    app_mean = self.appliance_params[appliance]['mean']
                    app_std = self.appliance_params[appliance]['std']
                    prediction = prediction * app_std + app_mean
                    valid_predictions = prediction.flatten()
                    valid_predictions[valid_predictions < 0] = 0
                    df = pd.Series(valid_predictions)
                    disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions