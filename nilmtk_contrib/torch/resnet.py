from __future__ import print_function, division

from nilmtk.disaggregate import Disaggregator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from collections import OrderedDict
from nilmtk_contrib.utils.validation import safe_train_test_split as train_test_split

# Set device
from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger

logger = module_logger(__name__)
_log_print = legacy_print(logger)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class IdentityBlock(nn.Module):
    """
    An identity block for ResNet, where the input and output dimensions are the same.
    This implementation mirrors the structure of the original TensorFlow version.
    """
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        
        # Three convolutional layers, maintaining the channel count
        self.conv1 = nn.Conv1d(in_channels=filters[0], out_channels=filters[0], 
                              kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], 
                              kernel_size=kernel_size, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=filters[1], out_channels=filters[2], 
                              kernel_size=kernel_size, stride=1, padding='same')
    
    def forward(self, x):
        # Store input for the residual connection
        identity = x
        
        # Forward pass through convolutions with ReLU activations
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        # Add the residual (identity) connection and apply final activation
        out += identity
        out = F.relu(out)
        
        return out

class ConvolutionBlock(nn.Module):
    """
    A convolutional block for ResNet that can change the input's channel dimension.
    This implementation mirrors the structure of the original TensorFlow version.
    """
    def __init__(self, filters, kernel_size):
        super(ConvolutionBlock, self).__init__()
        
        # Main path with three convolutional layers
        self.conv1 = nn.Conv1d(in_channels=filters[0], out_channels=filters[0], 
                              kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], 
                              kernel_size=kernel_size, stride=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=filters[1], out_channels=filters[2], 
                              kernel_size=kernel_size, stride=1, padding='same')
        
        # Skip connection path to match the output channel dimension
        self.conv4 = nn.Conv1d(in_channels=filters[0], out_channels=filters[2], 
                              kernel_size=kernel_size, stride=1, padding='same')
    
    def forward(self, x):
        # Store input for the skip connection
        identity = x
        
        # Forward pass through the main path
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        # Transform the identity to match the output channels for the residual connection
        identity = self.conv4(identity)
        
        # Add the residual connection and apply final activation
        out += identity
        out = F.relu(out)
        
        return out

class ResNetModel(nn.Module):
    """
    A ResNet-based model for NILM, mirroring the original TensorFlow implementation.
    """
    def __init__(self, sequence_length, num_filters=30):
        super(ResNetModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        
        # Initial layers, including double ReLU to match TensorFlow's structure
        self.zero_pad = nn.ZeroPad1d(3)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=48, stride=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        
        # ResNet blocks
        self.conv_block = ConvolutionBlock([num_filters, num_filters, num_filters], 24)
        self.identity_block1 = IdentityBlock([num_filters, num_filters, num_filters], 12)
        self.identity_block2 = IdentityBlock([num_filters, num_filters, num_filters], 6)
        
        # Calculate the input size for the fully connected layers dynamically
        self._calculate_fc_input_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, sequence_length)
    
    def _calculate_fc_input_size(self):
        """Calculates the input size for the FC layers via a dummy forward pass."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.sequence_length)
            x = self._forward_conv_layers(dummy_input)
            self.fc_input_size = x.flatten(1).shape[1]
    
    def _forward_conv_layers(self, x):
        """Performs the forward pass through the convolutional layers."""
        x = self.zero_pad(x)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.conv_block(x)
        x = self.identity_block1(x)
        x = self.identity_block2(x)
        
        return x
    
    def forward(self, x):
        # Convolutional layers
        x = self._forward_conv_layers(x)
        
        # Fully connected layers
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResNet(Disaggregator):
    """
    ResNet-based model for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385
    
    The model adapts the ResNet architecture for energy disaggregation tasks,
    using residual connections to enable training of deep networks for predicting
    individual appliance power consumption from aggregate household power measurements.
    
    Architecture Overview:
    - 1D convolutional layers adapted for time series data
    - Identity blocks with residual connections for feature learning
    - Convolution blocks for changing channel dimensions
    - Batch normalization and max pooling for regularization
    - Fully connected layers for sequence prediction
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 299)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
            - appliance_params (dict): Appliance-specific normalization parameters
            - load_model_path (str): Path to load pre-trained models
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        self.MODEL_NAME = "ResNet"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 299)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.load_model_path = params.get('load_model_path', None)
        self.appliance_params = params.get('appliance_params', {})
        self.device = device
        
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd!")
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        """Trains the model on a chunk of data."""
        _log_print("...............ResNet partial_fit running...............")
        
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)
        
        if do_preprocessing:
            _log_print("Preprocessing data...")
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        
        train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df_values = pd.concat(app_dfs, axis=0).values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        
        _log_print(f"Training data shape: {train_main.shape}")
        
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                _log_print(f"First time training for {appliance_name}")
                self.models[appliance_name] = self.return_network()
            else:
                _log_print(f"Retraining model for {appliance_name}")
            
            model = self.models[appliance_name]
            if train_main.size > 10:
                    # Create training and validation sets
                    train_x, v_x, train_y, v_y = train_test_split(
                        train_main, power, test_size=0.15, random_state=10)
                    
                    # Convert to PyTorch Tensors
                    train_x = torch.FloatTensor(train_x).permute(0, 2, 1).to(self.device)
                    v_x = torch.FloatTensor(v_x).permute(0, 2, 1).to(self.device)
                    train_y = torch.FloatTensor(train_y).to(self.device)
                    v_y = torch.FloatTensor(v_y).to(self.device)
                    
                    # Create DataLoaders for batching
                    train_dataset = TensorDataset(train_x, train_y)
                    val_dataset = TensorDataset(v_x, v_y)
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                    
                    # Train the model
                    self.train_model(model, train_loader, val_loader, appliance_name)
    
    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """
        Preprocesses data by windowing and normalizing, mirroring the original
        TensorFlow implementation.
        """
        if method == 'train':            
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                    self.appliance_params[app_name]['min']
                    self.appliance_params[app_name]['max']
                else:
                    raise ApplianceNotFoundError(f"Parameters for appliance '{app_name}' not found!")

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
                units_to_pad = n // 2
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst
    
    def train_model(self, model, train_loader, val_loader, appliance_name):
        """Handles the training and validation loop for the model."""
        # Optimizer with settings matching TensorFlow's defaults
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        _log_print(f"Training {appliance_name} for {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            # --- Training Phase ---
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Early stopping and saving the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                _log_print(f'Epoch {epoch+1}: New best model found with validation loss: {val_loss:.6f}')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                _log_print(f'Epoch {epoch+1}/{self.n_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Check for early stopping
            if patience_counter >= patience and epoch >= 20:
                _log_print(f"Stopping early at epoch {epoch+1} due to no improvement.")
                break
        
        # Load the best model state after training is complete
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            _log_print(f"Finished training. Loaded best model for {appliance_name} with validation loss: {best_val_loss:.6f}")
    
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregates a chunk of mains data."""
        if model is not None:
            self.models = model
        
        if do_preprocessing:
            _log_print("Preprocessing test data...")
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')
        
        test_predictions = []
        
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            test_main_tensor = torch.FloatTensor(test_main_array).permute(0, 2, 1).to(self.device)
            
            for appliance, model in self.models.items():
                model.eval()
                
                # Create DataLoader for batched predictions
                test_dataset = TensorDataset(test_main_tensor)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
                predictions = []
                with torch.no_grad():
                    for batch_x, in test_loader:
                        batch_pred = model(batch_x)
                        predictions.append(batch_pred.cpu().numpy())
                
                prediction = np.concatenate(predictions, axis=0)
                
                # Average predictions over overlapping windows
                window_length = self.sequence_length
                n = len(prediction) + window_length - 1
                sum_arr = np.zeros(n)
                counts_arr = np.zeros(n)
                
                for i, p in enumerate(prediction):
                    sum_arr[i:i+window_length] += p.flatten()
                    counts_arr[i:i+window_length] += 1
                
                # Replace zero counts with one to avoid division by zero
                counts_arr[counts_arr == 0] = 1
                averaged_prediction = sum_arr / counts_arr
                
                # Denormalize predictions
                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']
                denormalized_prediction = averaged_prediction * app_std + app_mean
                
                # Set negative values to zero
                denormalized_prediction[denormalized_prediction < 0] = 0
                df = pd.Series(denormalized_prediction)
                disggregation_dict[appliance] = df
            
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        
        return test_predictions
    
    def return_network(self):
        """Returns a new, initialized ResNet model."""
        model = ResNetModel(self.sequence_length).to(self.device)
        
        # Initialize weights to match TensorFlow's defaults
        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        return model
        
    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
        _log_print("Setting appliance parameters...")
        
        for (app_name, df_list) in train_appliances:
            values = np.concatenate([df.values for df in df_list])
            app_mean = np.mean(values)
            app_std = np.std(values)
            app_max = np.max(values)
            app_min = np.min(values)
            if app_std < 1:
                app_std = 100
            self.appliance_params[app_name] = {
                'mean': app_mean, 'std': app_std, 
                'max': app_max, 'min': app_min
            }
            _log_print(f"  {app_name}: mean={app_mean:.2f}, std={app_std:.2f}")
