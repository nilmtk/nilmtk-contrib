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

class ConvLSTM(Disaggregator):
    """
    Convolutional LSTM for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    https://arxiv.org/abs/1506.04214
    
    The model adapts the ConvLSTM architecture for energy disaggregation tasks,
    using spatiotemporal sequence modeling to predict individual appliance power consumption
    from aggregate household power measurements.
    
    Architecture Overview:
    - Convolutional LSTM layers for spatiotemporal feature learning
    - Dropout and dense layers for regularization and output prediction
    - Sequence-to-point prediction for energy disaggregation
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 99)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
            - appliance_params (dict): Appliance-specific normalization parameters
            - mains_mean (float): Mean value for mains normalization (default: 1800)
            - mains_std (float): Standard deviation for mains normalization (default: 600)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        super().__init__()
        self.MODEL_NAME = "ConvLSTM"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Extract legacy hyperparameters used by the Seq2Point-style training path.
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        self.mains_mean = params.get("mains_mean", 1800)
        self.mains_std = params.get("mains_std", 600)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sequence length must be odd for proper windowing
        if self.sequence_length % 2 == 0:
            _log_print("Sequence length should be odd!")
            raise SequenceLengthError

    def return_network(self):
        """
        Builds the Conv-LSTM network architecture.
        """
        class ConvLSTMNet(nn.Module):
            def __init__(self, sequence_length):
                super().__init__()
                
                # Convolutional feature extraction layers
                # Similar to seq2point but with fewer layers for LSTM compatibility
                self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=3)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=6, stride=1, padding=2)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1)
                
                # Calculate conv output length
                self.conv_output_dim = 128
                
                # Dropout for regularization
                self.dropout1 = nn.Dropout(0.2)
                
                # BiLSTM layers for temporal modeling
                self.lstm1 = nn.LSTM(
                    input_size=self.conv_output_dim,
                    hidden_size=128,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.0
                )
                
                self.lstm2 = nn.LSTM(
                    input_size=256,  # 128 * 2 (bidirectional)
                    hidden_size=64,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.0
                )
                
                self.dropout2 = nn.Dropout(0.2)
                
                # Final prediction layers
                self.fc1 = nn.Linear(128, 64)  # 64 * 2 (bidirectional)
                self.fc2 = nn.Linear(64, 1)
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                """
                Initializes model weights.
                """
                for m in self.modules():
                    if isinstance(m, nn.Conv1d):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight_ih' in name:
                                nn.init.xavier_uniform_(param.data)
                            elif 'weight_hh' in name:
                                nn.init.orthogonal_(param.data)
                            elif 'bias' in name:
                                nn.init.zeros_(param.data)
            
            def forward(self, x):
                # x shape: (batch_size, 1, sequence_length)
                
                # Convolutional feature extraction
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = self.dropout1(x)
                
                # Reshape for LSTM: (batch_size, sequence_length, features)
                x = x.transpose(1, 2)  # (batch_size, sequence_length, conv_output_dim)
                
                # BiLSTM layers
                x, _ = self.lstm1(x)
                x, _ = self.lstm2(x)
                x = self.dropout2(x)
                
                # Take the last timestep output for sequence-to-point prediction
                x = x[:, -1, :]  # (batch_size, hidden_size * 2)
                
                # Final prediction layers
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                
                return x
        
        model = ConvLSTMNet(self.sequence_length).to(self.device)
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """
        Preprocesses data by creating sliding windows, same as seq2point.
        """
        if method == 'train':
            # Preprocessing for the train data follows the Seq2Point-style path.
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
                    _log_print("Parameters for", app_name, "were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []
                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    # This is for choosing windows
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    # Return as a list of dataframe
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list
        
        else:
            # Preprocessing for the test data follows the Seq2Point-style path.
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
        """
        Computes and sets normalization parameters for each appliance.
        """
        for app_name, df_list in train_appliances:
            values = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
        _log_print(self.appliance_params)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """
        Trains the Conv-LSTM model on a chunk of data.
        """
        # If no appliance wise parameters are provided, then compute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        _log_print("...............ConvLSTM partial_fit running...............")
        # Do the pre-processing, such as windowing and normalizing
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
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                _log_print("First model training for", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                _log_print("Started Retraining model for", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Convert to PyTorch tensors and correct format
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
                    
                    # Setup optimizer and loss
                    optimizer = torch.optim.Adam(model.parameters())
                    criterion = nn.MSELoss()
                    
                    best_val_loss = float('inf')
                    filepath = checkpoint_path(".pth")
                    
                    # Training loop follows the Seq2Point-style behavior.
                    for epoch in range(self.n_epochs):
                        model.train()
                        
                        # Create batches
                        train_dataset = TensorDataset(train_X, train_y)
                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                        
                        epoch_losses = []
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            predictions = model(batch_X).squeeze()
                            loss = criterion(predictions, batch_y)
                            loss.backward()
                            
                            # Add gradient clipping like seq2point_new
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
                        
                        # Save best model using the legacy checkpoint behavior.
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), filepath)
                            _log_print(f"Validation loss improved, saving model to {filepath}")
                    
                    # Load best weights
                    model.load_state_dict(torch.load(filepath, map_location=self.device))

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """
        Disaggregates a chunk of mains power data.
        """
        if model is not None:
            self.models = model

        # Preprocess the test mains such as windowing and normalizing
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            
            # Convert to PyTorch tensor with correct format for Conv1d
            test_main_tensor = torch.tensor(test_main, dtype=torch.float32).permute(0, 2, 1).to(self.device)
            
            disggregation_dict = {}
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                with torch.no_grad():
                    prediction = model(test_main_tensor).cpu().numpy()
                    # Denormalize with the Seq2Point-style appliance parameters.
                    prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                    valid_predictions = prediction.flatten()
                    valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                    df = pd.Series(valid_predictions)
                    disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions
