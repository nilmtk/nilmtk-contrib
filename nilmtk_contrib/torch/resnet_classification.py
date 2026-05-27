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

class ResNetClassificationNet(nn.Module):
    """
    A ResNet-based network for NILM that combines a classification subnetwork
    and a regression subnetwork, mirroring the original TensorFlow implementation.
    """
    def __init__(self, sequence_length):
        super(ResNetClassificationNet, self).__init__()
        self.sequence_length = sequence_length
        
        # --- CLASSIFICATION SUBNETWORK ---
        self.cls_conv1 = nn.Conv1d(1, 30, kernel_size=10, padding='valid')
        self.cls_conv2 = nn.Conv1d(30, 30, kernel_size=8, padding='valid')
        self.cls_conv3 = nn.Conv1d(30, 40, kernel_size=6, padding='valid')
        self.cls_conv4 = nn.Conv1d(40, 50, kernel_size=5, padding='valid')
        self.cls_conv5 = nn.Conv1d(50, 50, kernel_size=5, padding='valid')
        self.cls_conv6 = nn.Conv1d(50, 50, kernel_size=5, padding='valid')
        
        # Calculate flattened size after convolutions
        conv_output_length = sequence_length - (10-1) - (8-1) - (6-1) - (5-1) - (5-1) - (5-1)
        self.cls_flatten_size = 50 * conv_output_length
        
        self.cls_dense1 = nn.Linear(self.cls_flatten_size, 1024)
        self.cls_dense2 = nn.Linear(1024, sequence_length)
        
        # --- REGRESSION SUBNETWORK (ResNet) ---
        self.zero_pad = nn.ZeroPad1d(3)
        self.reg_conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=48, stride=2)
        self.reg_bn1 = nn.BatchNorm1d(30)
        self.reg_maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        
        # ResNet blocks with parameters aligned to the TensorFlow backend.
        self.conv_block = ConvolutionBlock([30, 30, 30], 24)
        self.identity_block1 = IdentityBlock([30, 30, 30], 12)
        self.identity_block2 = IdentityBlock([30, 30, 30], 6)
        
        # Calculate the input size for the fully connected layers dynamically
        self._calculate_fc_input_size()
        
        # Fully connected layers for regression
        self.reg_fc1 = nn.Linear(self.fc_input_size, 1024)
        self.reg_dropout = nn.Dropout(0.2)
        self.reg_fc2 = nn.Linear(1024, sequence_length)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_fc_input_size(self):
        """Calculates the input size for the FC layers via a dummy forward pass."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.sequence_length)
            x = self._forward_regression_conv_layers(dummy_input)
            self.fc_input_size = x.flatten(1).shape[1]
    
    def _forward_regression_conv_layers(self, x):
        """Performs the forward pass through the regression conv layers."""
        x = self.zero_pad(x)
        x = F.relu(self.reg_conv1(x))
        x = self.reg_bn1(x)
        x = F.relu(x)
        x = self.reg_maxpool(x)
        
        x = self.conv_block(x)
        x = self.identity_block1(x)
        x = self.identity_block2(x)
        
        return x
    
    def _initialize_weights(self):
        """Initializes weights to match TensorFlow's defaults."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Use He normal initialization for the first dense layer in classification
        nn.init.kaiming_normal_(self.cls_dense1.weight, nonlinearity='relu')
    
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        
        # --- CLASSIFICATION SUBNETWORK ---
        cls_x = F.relu(self.cls_conv1(x))
        cls_x = F.relu(self.cls_conv2(cls_x))
        cls_x = F.relu(self.cls_conv3(cls_x))
        cls_x = F.relu(self.cls_conv4(cls_x))
        cls_x = F.relu(self.cls_conv5(cls_x))
        cls_x = F.relu(self.cls_conv6(cls_x))
        cls_x = cls_x.view(cls_x.size(0), -1)  # Flatten
        cls_x = F.relu(self.cls_dense1(cls_x))
        classification_output = torch.sigmoid(self.cls_dense2(cls_x))
        
        # --- REGRESSION SUBNETWORK ---
        reg_x = self._forward_regression_conv_layers(x)
        
        # Flatten and pass through dense layers
        reg_x = reg_x.flatten(1)
        reg_x = F.relu(self.reg_fc1(reg_x))
        reg_x = self.reg_dropout(reg_x)
        regression_output = self.reg_fc2(reg_x)
        
        # Final output is the element-wise product of the two subnetworks
        output = regression_output * classification_output
        
        return output, classification_output

class ResNet_classification(Disaggregator):
    """
    ResNet-based model with classification for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "ResNet-based Multi-output Regression for NILM: Towards Enhanced Appliance State Detection"
    https://arxiv.org/abs/2411.15805v1
    
    The model combines ResNet architecture with dual-output design for both appliance 
    state classification and power consumption regression in energy disaggregation tasks.
    
    Architecture Overview:
    - Classification subnetwork with 1D convolutions for appliance state detection
    - Regression subnetwork with ResNet blocks for power prediction
    - Identity and convolution blocks with residual connections
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
        self.MODEL_NAME = "ResNet_classification"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
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
        """Returns a new instance of the ResNetClassificationNet."""
        return ResNetClassificationNet(self.sequence_length).to(self.device)

    def classify(self, classify_appliance):
        """Creates binary on/off classification labels for appliances."""
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
                appliance_copy = appliance.copy()
                appliance_copy[appliance_copy <= threshold] = 0
                appliance_copy[appliance_copy > threshold] = 1
                new_app_readings = appliance_copy.values.flatten()
                new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                classification_appliance_dfs.append(pd.DataFrame(new_app_readings))
            appliance_on_off.append((appliance_name, classification_appliance_dfs))
        return appliance_on_off

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """Preprocesses data by windowing and normalizing."""
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
                    # Normalize using min-max scaling
                    new_app_readings = (new_app_readings - app_min) / (app_max - app_min)
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))

                appliance_list.append((app_name, processed_app_dfs))

            return processed_mains_lst, appliance_list

        else:
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
        values = np.concatenate([mains.values.flatten() for mains in train_main])
        self.mains_params.update({
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        })

    def set_appliance_params(self, train_appliances):
        """Computes and sets normalization parameters for each appliance."""
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
                'min': app_min, 'max': app_max
            }

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        """Trains the model on a chunk of data."""
        _log_print("...............ResNet_classification partial_fit running...............")
        
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)
        if not self.mains_params:
            self.set_mains_params(train_main)

        if do_preprocessing:
            # Create classification labels
            classify_appliance = copy.deepcopy(train_appliances)
            classification = self.classify(classify_appliance)

            # Preprocess regression and classification data
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        
        train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))

        # Process appliance data for regression
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df_values = pd.concat(app_dfs, axis=0).values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        # Process appliance data for classification
        new_train_appliances_classification = {}
        for app_name, app_df in classification:
            app_df_values = pd.concat(app_df, axis=0).values.reshape((-1, self.sequence_length))
            new_train_appliances_classification[app_name] = app_df_values
        
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                _log_print("First time training for", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                _log_print("Retraining model for", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 10:
                    # Combine regression and classification targets
                    power_df = pd.DataFrame(power)
                    classification_df = pd.DataFrame(new_train_appliances_classification[appliance_name])
                    power_combined = pd.concat([power_df, classification_df], axis=1).values

                    # Split data into training and validation sets
                    train_x, v_x, train_y_combined, v_y_combined = train_test_split(
                        train_main, power_combined, test_size=0.15, random_state=10)

                    train_y = train_y_combined[:, :self.sequence_length]
                    v_y = v_y_combined[:, :self.sequence_length]
                    appliance_train_classification = train_y_combined[:, self.sequence_length:]
                    appliance_val_classification = v_y_combined[:, self.sequence_length:]

                    # Convert to PyTorch tensors
                    train_x = torch.tensor(train_x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                    v_x = torch.tensor(v_x, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                    train_y = torch.tensor(train_y, dtype=torch.float32).to(self.device)
                    v_y = torch.tensor(v_y, dtype=torch.float32).to(self.device)
                    appliance_train_classification = torch.tensor(appliance_train_classification, dtype=torch.float32).to(self.device)
                    appliance_val_classification = torch.tensor(appliance_val_classification, dtype=torch.float32).to(self.device)

                    # Setup optimizer and loss functions
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                    mse_loss = nn.MSELoss()
                    bce_loss = nn.BCELoss()

                    best_val_loss = float('inf')
                    filepath = checkpoint_path(".pth")

                    # Training loop
                    for epoch in range(self.n_epochs):
                        model.train()
                        
                        train_dataset = TensorDataset(train_x, train_y, appliance_train_classification)
                        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

                        epoch_losses = []
                        for batch_x, batch_y, batch_c in train_loader:
                            optimizer.zero_grad()
                            output, classification_output = model(batch_x)
                            
                            # Combined loss for regression and classification
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
                            val_output, val_classification = model(v_x)
                            val_loss = (
                                self.regression_loss_weight * mse_loss(val_output, v_y)
                                + self.classification_loss_weight * bce_loss(val_classification, appliance_val_classification)
                            )

                        avg_train_loss = np.mean(epoch_losses)
                        _log_print(f"Epoch {epoch+1}/{self.n_epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss.item():.4f}")

                        # Save the best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), filepath)
                            _log_print(f"Validation loss improved, saving model to {filepath}")

                    # Load best weights
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
                    prediction_output, _ = model(test_main_tensor)
                    prediction = prediction_output.cpu().numpy()
                
                # Average predictions over overlapping windows
                window_length = self.sequence_length
                n = len(prediction)
                sum_arr = np.zeros(n + window_length - 1)
                counts_arr = np.zeros(n + window_length - 1)
                for i in range(n):
                    sum_arr[i:i+window_length] += prediction[i]
                    counts_arr[i:i+window_length] += 1
                for i in range(len(counts_arr)):
                    if counts_arr[i] == 0:
                        counts_arr[i] = 1
                averaged_prediction = sum_arr / counts_arr
                
                # Denormalize the predictions
                app_min = self.appliance_params[appliance]['min']
                app_max = self.appliance_params[appliance]['max']
                prediction = averaged_prediction * (app_max - app_min) + app_min
                prediction[prediction < 0] = 0
                
                df = pd.Series(prediction)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def classification_output_plot(self, prediction_classification, appliance):
        """Optional plotting function for classification output (matching TensorFlow)"""
        pass  # Placeholder for plotting functionality
