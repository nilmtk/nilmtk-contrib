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

# Use GPU if available, otherwise fall back to CPU
from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class AttentionLayer(nn.Module):
    """
    An attention mechanism that computes a context-aware representation of the input sequence.
    This implementation is designed to mirror the original TensorFlow version.
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        # Linear layers for computing attention scores
        self.W = nn.Linear(512, units)  # Input is from a bidirectional LSTM (256*2)
        self.V = nn.Linear(units, 1)
        
        # Initialize weights with He normal to match TensorFlow's 'he_normal'
        nn.init.kaiming_normal_(self.W.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.V.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.W.bias)
        nn.init.zeros_(self.V.bias)
    
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: The output from the LSTM layer, shape (batch, seq_len, hidden_size).
        Returns:
            context_vector: The weighted sum of encoder outputs, shape (batch, hidden_size).
        """
        # Calculate alignment scores
        score = self.V(torch.tanh(self.W(encoder_output)))  # (batch, seq_len, 1)
        
        # Convert scores to weights using softmax
        attention_weights = F.softmax(score, dim=1)
        
        # Compute the context vector
        context_vector = attention_weights * encoder_output
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector

class RNNAttentionModel(nn.Module):
    """
    An RNN-based model with an attention mechanism for NILM, designed to
    mirror the original TensorFlow implementation.
    """
    def __init__(self, sequence_length):
        super(RNNAttentionModel, self).__init__()
        self.sequence_length = sequence_length
        
        # Layers are defined to match the TensorFlow architecture
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, 
                                stride=1, padding=2) # 'same' padding
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(units=128)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        
        # Attention and final prediction
        x = self.attention(x)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x

class RNN_attention(Disaggregator):
    """
    RNN with attention mechanism for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "ResNet-based Multi-output Regression for NILM: Towards Enhanced Appliance State Detection"
    https://arxiv.org/abs/2411.15805v1
    
    The model uses bidirectional LSTM layers with attention mechanism for learning 
    temporal dependencies and focusing on relevant time steps in energy 
    disaggregation tasks.
    
    Architecture Overview:
    - Bidirectional LSTM layers for sequence modeling
    - Attention mechanism for learning relevant temporal features
    - Dense layers for final power consumption prediction
    - Sequence-to-point prediction for energy disaggregation
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 19)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
            - appliance_params (dict): Appliance-specific normalization parameters
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        """Initializes the disaggregator and its hyperparameters."""
        self.MODEL_NAME = "RNN_attention"
        self.models = OrderedDict()
        
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 19)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.load_model_path = params.get('load_model_path', None)
        self.appliance_params = params.get('appliance_params', {})
        self.mains_mean = params.get('mains_mean', 1800)
        self.mains_std = params.get('mains_std', 600)
        self.device = device
        
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd for proper windowing.")
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        """Trains the model on a chunk of data."""
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)
        
        _log_print("...............RNN_attention partial_fit running...............")
        
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
        
        # Train a model for each appliance
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
                train_x = torch.FloatTensor(train_x).to(self.device)
                v_x = torch.FloatTensor(v_x).to(self.device)
                train_y = torch.FloatTensor(train_y).to(self.device)
                v_y = torch.FloatTensor(v_y).to(self.device)
                
                # Create DataLoaders
                train_dataset = TensorDataset(train_x, train_y)
                val_dataset = TensorDataset(v_x, v_y)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                
                self.train_model(model, train_loader, val_loader, appliance_name)
    
    def train_model(self, model, train_loader, val_loader, appliance_name):
        """Handles the training and validation loop for a single appliance model."""
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.n_epochs):
            # --- Training Phase ---
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # --- Validation Phase ---
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                
                filepath = checkpoint_path(".pth")
                torch.save(best_model_state, filepath)
                _log_print(f'Epoch {epoch+1}: val_loss improved to {val_loss:.6f}, saving model to {filepath}')
        
        # Load the best performing model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
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
            test_main_tensor = torch.FloatTensor(test_main_array).to(self.device)
            
            disggregation_dict = {}
            
            for appliance, model in self.models.items():
                model.eval()
                
                # Create DataLoader for batched inference
                test_dataset = TensorDataset(test_main_tensor)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
                predictions = []
                with torch.no_grad():
                    for batch_x, in test_loader:
                        batch_pred = model(batch_x)
                        predictions.append(batch_pred.cpu().numpy())
                
                prediction = np.concatenate(predictions, axis=0)
                
                # Denormalize predictions
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
        """Returns a new, initialized RNNAttentionModel instance."""
        model = RNNAttentionModel(self.sequence_length).to(self.device)
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
