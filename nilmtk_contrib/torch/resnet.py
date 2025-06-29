from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from nilmtk_contrib.torch.preprocessing import preprocess

# Set random seeds
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class IdentityBlock(nn.Module):
    def __init__(self, filters, kernel_size, input_channels=None):
        super(IdentityBlock, self).__init__()
        
        # Use input_channels if provided, otherwise assume filters[0]
        in_channels = input_channels if input_channels is not None else filters[0]
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=filters[0], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(in_channels=filters[1], out_channels=filters[2], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # Shortcut connection - adjust if input and output channels don't match
        if in_channels != filters[2]:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=filters[2], 
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        identity = self.shortcut(identity)
        
        # Ensure both tensors have the same size
        if out.size() != identity.size():
            # Adjust size if needed
            min_size = min(out.size(2), identity.size(2))
            out = out[:, :, :min_size]
            identity = identity[:, :, :min_size]
        
        out = out + identity
        out = F.relu(out)
        
        return out

class ConvolutionBlock(nn.Module):
    def __init__(self, filters, kernel_size, input_channels=None):
        super(ConvolutionBlock, self).__init__()
        
        # Use input_channels if provided, otherwise assume filters[0]
        in_channels = input_channels if input_channels is not None else filters[0]
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=filters[0], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(in_channels=filters[1], out_channels=filters[2], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv4 = nn.Conv1d(in_channels=in_channels, out_channels=filters[2], 
                              kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        
        identity = F.relu(self.conv4(identity))
        
        # Ensure both tensors have the same size
        if out.size() != identity.size():
            min_size = min(out.size(2), identity.size(2))
            out = out[:, :, :min_size]
            identity = identity[:, :, :min_size]
        
        out = out + identity
        out = F.relu(out)
        
        return out

class ResNetModel(nn.Module):
    """
    ResNet model for appliance load disaggregation.
    It includes initial convolutional layers, ResNet blocks, and fully connected layers.
    """
    def __init__(self, sequence_length, num_filters=30):
        super(ResNetModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        
        # Initial layers - matching TensorFlow implementation exactly
        self.zero_pad = nn.ZeroPad1d(3)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, 
                              kernel_size=48, stride=2, padding=0)  # No padding here, ZeroPad1d handles it
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        
        # Calculate intermediate size after initial layers
        self._calculate_intermediate_size()
        
        # ResNet blocks with proper input channel specification
        self.conv_block = ConvolutionBlock([num_filters, num_filters, num_filters], 24, 
                                         input_channels=num_filters)
        self.identity_block1 = IdentityBlock([num_filters, num_filters, num_filters], 12,
                                           input_channels=num_filters)
        self.identity_block2 = IdentityBlock([num_filters, num_filters, num_filters], 6,
                                           input_channels=num_filters)
        
        # Calculate the size after convolutions for fully connected layers
        self._calculate_fc_input_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, sequence_length)
    
    def _calculate_intermediate_size(self):
        """Calculate size after initial conv and maxpool layers"""
        # Start with sequence_length + 6 (3 padding on each side)
        size = self.sequence_length + 6
        # After conv1 with kernel=48, stride=2
        size = (size - 48) // 2 + 1
        # After maxpool with kernel=3, stride=2  
        size = (size - 3) // 2 + 1
        self.intermediate_size = size
    
    def _calculate_fc_input_size(self):
        """Calculate the size after all convolutions"""
        # Create a dummy input to calculate the size after convolutions
        dummy_input = torch.zeros(1, 1, self.sequence_length)
        x = self._forward_conv_layers(dummy_input)
        x = x.view(x.size(0), -1)
        self.fc_input_size = x.size(1)
    
    def _forward_conv_layers(self, x):
        """Forward pass through convolutional layers only"""
        # Initial processing
        x = self.zero_pad(x)
        x = self.conv1(x)
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
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResNet(Disaggregator):
    """
    ResNet-based disaggregator for NILMTK.
    This class implements a ResNet model for disaggregating mains electricity data
    into appliance-level data.
    """ 
    def __init__(self, params):
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
            print("Sequence length should be odd!")
            raise SequenceLengthError
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        print("...............ResNet partial_fit running...............")
        
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        
        if do_preprocessing:
            print("Preprocessing data...")
            train_main, train_appliances = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=train_main,
                submeters_lst=train_appliances,
                method="train",
                appliance_params=self.appliance_params,
                windowing=True
            )
        
        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        
        print(f"Training data shape: {train_main.shape}")
        
        # Progress bar for appliances
        appliance_progress = tqdm(train_appliances, desc="Training appliances", unit="appliance")
        
        for appliance_name, power in appliance_progress:
            appliance_progress.set_postfix({"Current": appliance_name})
            
            if appliance_name not in self.models:
                print(f"\nFirst model training for {appliance_name}")
                self.models[appliance_name] = self.return_network()
            else:
                print(f"\nStarted Retraining model for {appliance_name}")
            
            model = self.models[appliance_name]
            if train_main.size > 0:
                if len(train_main) > 10:
                    # Convert to PyTorch tensors
                    train_x, v_x, train_y, v_y = train_test_split(
                        train_main, power, test_size=.15, random_state=10)
                    
                    train_x = torch.FloatTensor(train_x).permute(0, 2, 1).to(self.device)
                    v_x = torch.FloatTensor(v_x).permute(0, 2, 1).to(self.device)
                    train_y = torch.FloatTensor(train_y).to(self.device)
                    v_y = torch.FloatTensor(v_y).to(self.device)
                    
                    # Create DataLoaders
                    train_dataset = TensorDataset(train_x, train_y)
                    val_dataset = TensorDataset(v_x, v_y)
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                    
                    # Training loop
                    self.train_model(model, train_loader, val_loader, appliance_name)
    
    def train_model(self, model, train_loader, val_loader, appliance_name):
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        # Progress bar for epochs
        epoch_progress = tqdm(range(self.n_epochs), desc=f"Training {appliance_name}", unit="epoch")
        
        for epoch in epoch_progress:
            # Training phase
            model.train()
            train_loss = 0.0
            
            # Progress bar for training batches
            train_batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", 
                                      leave=False, unit="batch")
            
            for batch_x, batch_y in train_batch_progress:
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batch_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            # Progress bar for validation batches
            val_batch_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", 
                                    leave=False, unit="batch")
            
            with torch.no_grad():
                for batch_x, batch_y in val_batch_progress:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batch_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best": f"{best_val_loss:.4f}"
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epoch_progress.write(f'New best model saved with val_loss: {val_loss:.4f}')
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nLoaded best model for {appliance_name} with validation loss: {best_val_loss:.4f}")
    
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model
        
        if do_preprocessing:
            print("Preprocessing test data...")
            test_main_list = preprocess(
                sequence_length=self.sequence_length,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                mains_lst=test_main_list,
                submeters_lst=None,
                method="test",
                appliance_params=self.appliance_params,
                windowing=True
            )
        
        test_predictions = []
        
        # Progress bar for test chunks
        chunk_progress = tqdm(test_main_list, desc="Processing test chunks", unit="chunk")
        
        for test_mains_df in chunk_progress:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            test_main_tensor = torch.FloatTensor(test_main_array).permute(0, 2, 1).to(self.device)
            
            # Progress bar for appliances in each chunk
            appliance_progress = tqdm(self.models.items(), desc="Disaggregating appliances", 
                                    leave=False, unit="appliance")
            
            for appliance, model in appliance_progress:
                appliance_progress.set_postfix({"Current": appliance})
                
                model.eval()
                
                # Create DataLoader for batched prediction
                test_dataset = TensorDataset(test_main_tensor)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
                predictions = []
                
                # Progress bar for prediction batches
                pred_progress = tqdm(test_loader, desc=f"Predicting {appliance}", 
                                   leave=False, unit="batch")
                
                with torch.no_grad():
                    for batch_x, in pred_progress:
                        batch_pred = model(batch_x)
                        predictions.append(batch_pred.cpu().numpy())
                
                prediction = np.concatenate(predictions, axis=0)
                
                # Average predictions over sequences
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]
                
                # Denormalize predictions
                prediction = (self.appliance_params[appliance]['mean'] + 
                            (sum_arr * self.appliance_params[appliance]['std']))
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        
        return test_predictions
    
    def return_network(self):
        model = ResNetModel(self.sequence_length).to(self.device)
        return model
        
    def set_appliance_params(self, train_appliances):
        print("Setting appliance parameters...")
        
        # Progress bar for setting appliance parameters
        param_progress = tqdm(train_appliances, desc="Computing appliance stats", unit="appliance")
        
        for (app_name, df_list) in param_progress:
            param_progress.set_postfix({"Current": app_name})
            
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            app_max = np.max(l)
            app_min = np.min(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std, 
                                                   'max': app_max, 'min': app_min}})