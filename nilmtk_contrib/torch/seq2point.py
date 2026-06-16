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

class Seq2PointTorch(Disaggregator):
    """
    Sequence-to-Point neural network for Non-Intrusive Load Monitoring (NILM).
    
    Based on "Sequence-to-Point Learning With Neural Networks for Non-Intrusive Load Monitoring"
    by Zhang et al., published in Proceedings of the AAAI Conference on Artificial Intelligence, 2018.
    DOI: https://doi.org/10.1609/aaai.v32i1.11873
    
    This model uses a sequence-to-point learning approach where the input is a window 
    of mains power consumption and the output is a single point prediction of the target 
    appliance power. The architecture uses convolutional neural networks that can inherently 
    learn appliance signatures to reduce the identifiability problem in energy disaggregation.
    
    Architecture Overview:
    - Multiple 1D convolutional layers for feature extraction from power sequences
    - Dropout layer for regularization
    - Fully connected layers for final power prediction
    - Single point output from sequence input (sequence-to-point learning)
    
    Args:
        params (dict): Dictionary containing model hyperparameters:
            - sequence_length (int): Length of input sequences (default: 99, must be odd)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - appliance_params (dict): Appliance-specific normalization parameters
            - mains_mean (float): Mean normalization for mains power (default: 1800)
            - mains_std (float): Standard deviation for mains power (default: 600)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        """Initializes the disaggregator and its hyperparameters."""
        super().__init__()
        self.MODEL_NAME = "Seq2PointTorch"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        self.mains_mean = params.get("mains_mean", 1800)
        self.mains_std = params.get("mains_std", 600)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd for proper windowing.")

    def return_network(self):
        """Builds the 1D CNN model, mirroring the original TensorFlow architecture."""
        class Seq2PointNet(nn.Module):
            """The Seq2Point neural network architecture."""
            def __init__(self, sequence_length):
                super().__init__()
                # Layer definitions to match the original TensorFlow model
                self.conv1 = nn.Conv1d(1, 30, kernel_size=10, stride=1)
                self.conv2 = nn.Conv1d(30, 30, kernel_size=8, stride=1)
                self.conv3 = nn.Conv1d(30, 40, kernel_size=6, stride=1)
                self.conv4 = nn.Conv1d(40, 50, kernel_size=5, stride=1)
                self.conv5 = nn.Conv1d(50, 50, kernel_size=5, stride=1)
                self.dropout = nn.Dropout(0.2)
                
                # Calculate the flattened size dynamically after convolutions
                self._calculate_flatten_size(sequence_length)
                
                self.fc1 = nn.Linear(self.flatten_size, 1024)
                self.fc2 = nn.Linear(1024, 1)
                
                self._initialize_weights()

            def _calculate_flatten_size(self, seq_len):
                """Calculates the input size for the fully connected layer."""
                # Each conv layer reduces length by (kernel_size - 1)
                conv_output_length = seq_len - (10-1) - (8-1) - (6-1) - (5-1) - (5-1)
                self.flatten_size = 50 * conv_output_length
            
            def _initialize_weights(self):
                """Initializes weights to match TensorFlow's default (glorot_uniform)."""
                for m in self.modules():
                    if isinstance(m, (nn.Conv1d, nn.Linear)):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            def forward(self, x):
                # Forward pass through the network
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = torch.relu(self.conv4(x))
                x = self.dropout(x)
                x = torch.relu(self.conv5(x))
                x = self.dropout(x)
                x = x.flatten(1) # Flatten the output for the dense layers
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        model = Seq2PointNet(self.sequence_length).to(self.device)
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
        for app_name, df_list in train_appliances:
            values = np.concatenate([df.values for df in df_list])
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100 # Avoid division by zero for flat signals
            self.appliance_params[app_name] = {'mean': app_mean, 'std': app_std}
        _log_print("Appliance parameters set:", self.appliance_params)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """Trains the model on a chunk of data."""
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        _log_print("...............Seq2Point partial_fit running...............")
        
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
                    # PyTorch Conv1d expects (batch, channels, length)
                    train_main_tensor = torch.tensor(train_main, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                    power_tensor = torch.tensor(power, dtype=torch.float32).squeeze().to(self.device)
                    
                    # Create validation split
                    n_samples = train_main_tensor.size(0)
                    val_size = max(1, int(0.15 * n_samples)) if n_samples > 1 else 0
                    indices = torch.randperm(n_samples)
                    train_idx, val_idx = indices[val_size:], indices[:val_size]
                    
                    train_X = train_main_tensor[train_idx]
                    train_y = power_tensor[train_idx]
                    val_X = train_main_tensor[val_idx]
                    val_y = power_tensor[val_idx]
                    
                    # Optimizer and loss function
                    optimizer = torch.optim.Adam(model.parameters())
                    criterion = nn.MSELoss()
                    
                    best_val_loss = float('inf')
                    filepath = checkpoint_path(".pth")
                    
                    # Training loop
                    for epoch in range(self.n_epochs):
                        model.train()
                        
                        train_dataset = TensorDataset(train_X, train_y)
                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                        
                        epoch_losses = []
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            predictions = model(batch_X).squeeze()
                            loss = criterion(predictions, batch_y)
                            loss.backward()
                            
                            # Gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            epoch_losses.append(loss.item())
                        
                        # Validation
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
                    
                    # Load the best performing model
                    model.load_state_dict(torch.load(filepath, map_location=self.device))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregates a chunk of mains data."""
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            
            # PyTorch Conv1d expects (batch, channels, length)
            test_main_tensor = torch.tensor(test_main_array, dtype=torch.float32).permute(0, 2, 1).to(self.device)
            
            disggregation_dict = {}
            for appliance, model in self.models.items():
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_tensor).cpu().numpy()
                    
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