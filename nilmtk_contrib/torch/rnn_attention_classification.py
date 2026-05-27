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
import copy

# Set device
from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path
from nilmtk_contrib.preprocessing.classification import (
    appliance_threshold,
    classification_metadata,
    loss_weight_metadata,
)

logger = module_logger(__name__)
_log_print = legacy_print(logger)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class AttentionLayer(nn.Module):
    """
    An attention layer that computes a context vector from encoder outputs.
    This implementation is designed to mirror the original TensorFlow version.
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        # Layers to compute attention scores
        self.W = nn.Linear(units * 2, units)  # Input is bidirectional, hence *2
        self.V = nn.Linear(units, 1)
        
        # Initialize weights with He normal to match TensorFlow's default
        nn.init.kaiming_normal_(self.W.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.V.weight, nonlinearity='relu')
        nn.init.zeros_(self.W.bias)
        nn.init.zeros_(self.V.bias)
    
    def forward(self, encoder_output):
        """
        Args:
            encoder_output: The output from the LSTM layer, shape (batch, seq_len, hidden_size*2).
        Returns:
            context_vector: The weighted sum of encoder outputs, shape (batch, hidden_size*2).
            attention_weights: The computed attention weights, shape (batch, seq_len).
        """
        # Calculate alignment scores
        score = self.V(torch.tanh(self.W(encoder_output)))  # (batch, seq_len, 1)
        
        # Convert scores to weights using softmax
        attention_weights = F.softmax(score, dim=1)  # (batch, seq_len, 1)
        
        # Compute the context vector
        context_vector = attention_weights * encoder_output
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)

class RNNAttentionClassificationNet(nn.Module):
    """
    A dual-subnetwork model for NILM, combining a CNN-based classification
    network and an RNN-with-attention regression network. The architecture
    is designed to mirror the original TensorFlow implementation.
    """
    def __init__(self, sequence_length):
        super(RNNAttentionClassificationNet, self).__init__()
        self.sequence_length = sequence_length
        
        # --- CLASSIFICATION SUBNETWORK (CNN) ---
        self.cls_conv1 = nn.Conv1d(1, 30, kernel_size=10, padding='valid')
        self.cls_conv2 = nn.Conv1d(30, 30, kernel_size=8, padding='valid')
        self.cls_conv3 = nn.Conv1d(30, 40, kernel_size=6, padding='valid')
        self.cls_conv4 = nn.Conv1d(40, 50, kernel_size=5, padding='valid')
        self.cls_conv5 = nn.Conv1d(50, 50, kernel_size=5, padding='valid')
        self.cls_conv6 = nn.Conv1d(50, 50, kernel_size=5, padding='valid')
        
        # Calculate the flattened size dynamically after convolutions
        self._calculate_cls_flatten_size(sequence_length)
        
        self.cls_dense1 = nn.Linear(self.cls_flatten_size, 1024)
        self.cls_dense2 = nn.Linear(1024, sequence_length)
        
        # --- REGRESSION SUBNETWORK (RNN with Attention) ---
        self.reg_conv = nn.Conv1d(1, 16, kernel_size=4, stride=1, padding='same')
        self.bi_lstm1 = nn.LSTM(16, 128, batch_first=True, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(256)
        self.reg_dense1 = nn.Linear(512, 128)  # 512 = 256 * 2 (bidirectional)
        self.reg_dense2 = nn.Linear(128, sequence_length)
        
        self._initialize_weights()

    def _calculate_cls_flatten_size(self, seq_len):
        """Calculates the input size for the classification FC layer."""
        # Each conv layer reduces length by (kernel_size - 1)
        conv_output_length = seq_len - (10-1) - (8-1) - (6-1) - (5-1) - (5-1) - (5-1)
        self.cls_flatten_size = 50 * conv_output_length
    
    def _initialize_weights(self):
        """Initializes weights to match TensorFlow's default initializations."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # Use Xavier uniform for Conv and Linear layers by default
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                # Initialize LSTM weights and biases
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Performs the forward pass, combining classification and regression outputs.
        
        Args:
            x: Input tensor of shape (batch_size, 1, sequence_length).
        Returns:
            output: The final disaggregated power, shape (batch, seq_len).
            classification_output: The appliance status prediction, shape (batch, seq_len).
            attention_weights: The attention weights from the regression subnetwork, shape (batch, seq_len).
        """
        # --- CLASSIFICATION SUBNETWORK ---
        cls_x = F.relu(self.cls_conv1(x))
        cls_x = F.relu(self.cls_conv2(cls_x))
        cls_x = F.relu(self.cls_conv3(cls_x))
        cls_x = F.relu(self.cls_conv4(cls_x))
        cls_x = F.relu(self.cls_conv5(cls_x))
        cls_x = F.relu(self.cls_conv6(cls_x))
        cls_x = cls_x.flatten(1)
        cls_x = F.relu(self.cls_dense1(cls_x))
        classification_output = torch.sigmoid(self.cls_dense2(cls_x))
        
        # --- REGRESSION SUBNETWORK ---
        reg_x = self.reg_conv(x).permute(0, 2, 1)  # (batch, seq_len, 16)
        reg_x, _ = self.bi_lstm1(reg_x)
        reg_x, _ = self.bi_lstm2(reg_x)
        context_vector, attention_weights = self.attention(reg_x)
        reg_x = torch.tanh(self.reg_dense1(context_vector))
        regression_output = self.reg_dense2(reg_x)
        
        # Final output is the element-wise product of the two subnetworks
        output = regression_output * classification_output
        
        return output, classification_output, attention_weights

class RNN_attention_classification(Disaggregator):
    """
    RNN with attention and classification for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "ResNet-based Multi-output Regression for NILM: Towards Enhanced Appliance State Detection"
    https://arxiv.org/abs/2411.15805v1
    
    The model combines RNN with attention mechanism and CNN-based classification for 
    enhanced appliance state detection and power consumption prediction in energy 
    disaggregation tasks.
    
    Architecture Overview:
    - Classification subnetwork with 1D convolutions for appliance state detection
    - Regression subnetwork with bidirectional LSTM and attention mechanism
    - Attention layer for learning relevant temporal features
    - Element-wise multiplication of classification and regression outputs
    - Multi-output learning for enhanced appliance state detection
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 99)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
            - appliance_params (dict): Appliance-specific normalization parameters
            - mains_params (dict): Mains-specific normalization parameters
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        self.MODEL_NAME = "RNN_attention_classification"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.att_models = OrderedDict()  # Store attention models separately like TensorFlow
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        self.mains_params = params.get('mains_params', {})
        self.device = device
        self.classification_threshold = params.get('classification_threshold', params.get('on_power_threshold', 15))
        self.regression_loss_weight = params.get('regression_loss_weight', 1.0)
        self.classification_loss_weight = params.get('classification_loss_weight', 1.0)
        self.classification_metadata = classification_metadata(
            self.appliance_params,
            self.classification_threshold,
        )
        self.loss_weight_metadata = loss_weight_metadata(
            self.regression_loss_weight,
            self.classification_loss_weight,
        )
        
        if self.sequence_length % 2 == 0:
            raise SequenceLengthError("Sequence length must be odd!")

    def return_network(self):
        """Returns a new model and a corresponding attention model wrapper."""
        model = RNNAttentionClassificationNet(self.sequence_length).to(self.device)
        
        # Wrapper to extract attention weights, for compatibility with TF version
        class AttentionWrapper(nn.Module):
            def __init__(self, full_model):
                super().__init__()
                self.full_model = full_model
            
            def forward(self, x):
                _, _, attention_weights = self.full_model(x)
                return attention_weights
        
        attention_model = AttentionWrapper(model).to(self.device)
        return model, attention_model

    def classify(self, classify_appliance):
        """
        Generates binary on/off classification targets from appliance data.
        This preprocessing mirrors the original TensorFlow implementation.
        """
        appliance_on_off = []

        for app_index, (appliance_name, on_off_list) in enumerate(classify_appliance):
            threshold = appliance_threshold(
                self.appliance_params,
                appliance_name,
                self.classification_threshold,
            )
            classification_appliance_dfs = []
            for appliance in on_off_list:
                n = self.sequence_length
                units_to_pad = n // 2
                
                # Apply thresholding
                appliance_copy = appliance.copy()
                appliance_copy[appliance_copy <= threshold] = 0
                appliance_copy[appliance_copy > threshold] = 1
                
                # Create sequences
                new_app_readings = appliance_copy.values.flatten()
                new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                classification_appliance_dfs.append(pd.DataFrame(new_app_readings))
                
            appliance_on_off.append((appliance_name, classification_appliance_dfs))
        return appliance_on_off

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
                if app_name in self.appliance_params:
                    self.appliance_params[app_name]['mean']
                    self.appliance_params[app_name]['std']
                    app_min = self.appliance_params[app_name]['min']
                    app_max = self.appliance_params[app_name]['max']
                else:
                    raise ApplianceNotFoundError(f"Parameters for appliance '{app_name}' not found!")

                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                    # Normalize with min-max scaling, matching TensorFlow
                    new_app_readings = (new_app_readings - app_min) / (app_max - app_min)
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

    def set_mains_params(self, train_main):
        """Computes and sets normalization parameters for the mains data."""
        all_mains_data = np.concatenate([mains.values.flatten() for mains in train_main])
        self.mains_params = {
            'mean': np.mean(all_mains_data),
            'std': np.std(all_mains_data),
            'min': np.min(all_mains_data),
            'max': np.max(all_mains_data)
        }

    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
        for (app_name, df_list) in train_appliances:
            app_data = np.concatenate([df.values for df in df_list])
            app_mean = np.mean(app_data)
            app_std = np.std(app_data)
            if app_std < 1:
                app_std = 100  # Avoid division by zero for flat signals
            self.appliance_params[app_name] = {
                'mean': app_mean,
                'std': app_std,
                'min': np.min(app_data),
                'max': np.max(app_data)
            }

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        """Trains the model on a chunk of data."""
        _log_print("...............RNN_attention_classification partial_fit running...............")
        
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)
        if not self.mains_params:
            self.set_mains_params(train_main)

        if do_preprocessing:
            # Create classification targets before normalizing appliance data
            classify_appliance = copy.deepcopy(train_appliances)
            classification = self.classify(classify_appliance)
            
            # Normalize mains and appliance data
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        
        # Reshape all data into sequences
        train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))

        # Process appliance power data
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df_values = pd.concat(app_dfs, axis=0).values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        # Process classification target data
        new_train_appliances_classification = {}
        for app_name, app_dfs in classification:
            app_df_values = pd.concat(app_dfs, axis=0).values.reshape((-1, self.sequence_length))
            new_train_appliances_classification[app_name] = app_df_values
        
        self.att_models = {}
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                _log_print(f"First time training for {appliance_name}")
                self.models[appliance_name], self.att_models[appliance_name] = self.return_network()
            else:
                _log_print(f"Retraining model for {appliance_name}")

            model = self.models[appliance_name]
            if train_main.size > 10:
                    # Combine power and classification targets for splitting
                    power_classification_target = np.concatenate(
                        (power, new_train_appliances_classification[appliance_name]), axis=1)

                    # Create training and validation sets
                    train_x, v_x, train_y_combined, v_y_combined = train_test_split(
                        train_main, power_classification_target, test_size=0.15, random_state=10)

                    # Separate power and classification targets after splitting
                    train_y = train_y_combined[:, :self.sequence_length]
                    v_y = v_y_combined[:, :self.sequence_length]
                    train_c = train_y_combined[:, self.sequence_length:]
                    v_c = v_y_combined[:, self.sequence_length:]

                    # Convert to PyTorch Tensors
                    train_x = torch.tensor(train_x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                    v_x = torch.tensor(v_x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                    train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
                    v_y = torch.tensor(v_y, dtype=torch.float32).to(self.device)
                    train_c = torch.tensor(train_c, dtype=torch.float32).to(self.device)
                    v_c = torch.tensor(v_c, dtype=torch.float32).to(self.device)

                    # Optimizer and loss functions, matching TensorFlow
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                    mse_loss = nn.MSELoss()
                    bce_loss = nn.BCELoss()

                    best_val_loss = float('inf')
                    filepath = checkpoint_path(".pth")

                    # Training loop
                    for epoch in range(self.n_epochs):
                        model.train()
                        train_dataset = TensorDataset(train_x, train_y, train_c)
                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

                        epoch_losses = []
                        for batch_x, batch_y, batch_c in train_loader:
                            optimizer.zero_grad()
                            output, classification_output, _ = model(batch_x)
                            
                            # Combined loss (regression + classification)
                            loss = (
                                self.regression_loss_weight * mse_loss(output, batch_y)
                                + self.classification_loss_weight * bce_loss(classification_output, batch_c)
                            )
                            
                            loss.backward()
                            optimizer.step()
                            epoch_losses.append(loss.item())

                        # Validation
                        model.eval()
                        with torch.no_grad():
                            val_output, val_classification, _ = model(v_x)
                            val_loss = (
                                self.regression_loss_weight * mse_loss(val_output, v_y)
                                + self.classification_loss_weight * bce_loss(val_classification, v_c)
                            )

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
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            test_main_tensor = torch.tensor(test_main_array, dtype=torch.float32).permute(0, 2, 1).to(self.device)

            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                
                with torch.no_grad():
                    prediction_output, _, _ = model(test_main_tensor)
                    prediction_output = prediction_output.cpu().numpy()
                
                # Average predictions over overlapping windows to get a single series
                window_length = self.sequence_length
                n = len(prediction_output) + window_length - 1
                sum_arr = np.zeros(n)
                counts_arr = np.zeros(n)
                
                for i, p in enumerate(prediction_output):
                    sum_arr[i:i+window_length] += p.flatten()
                    counts_arr[i:i+window_length] += 1
                
                # Avoid division by zero
                counts_arr[counts_arr == 0] = 1
                averaged_prediction = sum_arr / counts_arr

                # Denormalize the prediction
                app_min = self.appliance_params[appliance]['min']
                app_max = self.appliance_params[appliance]['max']
                denormalized_prediction = app_min + (averaged_prediction * (app_max - app_min))
                
                # Set negative values to zero
                denormalized_prediction[denormalized_prediction < 0] = 0
                df = pd.Series(denormalized_prediction)
                disggregation_dict[appliance] = df

            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions
