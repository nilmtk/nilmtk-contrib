from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import os
from nilmtk_contrib.torch.preprocessing import preprocess

# Set random seeds for reproducibility across runs
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)

# Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class RNNModel(nn.Module):
    """
    Neural network combining CNN feature extraction and bidirectional LSTMs
    for NILM energy disaggregation.
    """
    def __init__(self, sequence_length):
        super(RNNModel, self).__init__()
        self.sequence_length = sequence_length
        
        # 1D CNN for initial feature extraction from raw power sequence
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=4, 
            stride=1, 
            padding=2  # Maintain sequence length
        )
        
        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=16,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer for deeper feature learning
        self.lstm2 = nn.LSTM(
            input_size=256,  # 128 * 2 (bidirectional)
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Final fully connected layers for prediction
        self.fc1 = nn.Linear(512, 128)  # 256 * 2 (bidirectional)
        self.fc2 = nn.Linear(128, 1)   # Output single power value
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, 1)
        # Rearrange for Conv1D: (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, 1, sequence_length)
        
        # Extract features using 1D convolution
        x = self.conv1d(x)  # (batch_size, 16, sequence_length)
        
        # Rearrange back for LSTM: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 16)
        
        # Process through bidirectional LSTM layers
        x, _ = self.lstm1(x)  # (batch_size, sequence_length, 256)
        x = self.dropout(x)
        
        x, _ = self.lstm2(x)  # (batch_size, sequence_length, 512)
        
        # Use only the last time step output
        x = x[:, -1, :]  # (batch_size, 512)
        
        # Final prediction layers
        x = torch.tanh(self.fc1(x))  # (batch_size, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 1)
        
        return x

class RNN(Disaggregator):
    """
    NILM disaggregator using RNN without attention mechanism.
    Inherits from NILMTK's Disaggregator base class.
    """
    
    def __init__(self, params):
        """Initialize the disaggregator with hyperparameters"""
        self.MODEL_NAME = "RNN"
        self.models = OrderedDict()  # Store separate models for each appliance
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        
        # Extract hyperparameters from params dict
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 19)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})  # Normalization stats
        self.mains_mean = params.get('mains_mean', 1800)
        self.mains_std = params.get('mains_std', 600)
        self.device = device
        
        # Sequence length must be odd for proper windowing
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """Train models on a chunk of data (supports incremental learning)"""
        
        # Compute appliance-specific normalization parameters if not provided
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        
        print("...............RNN partial_fit running...............")
        
        # Preprocess data: windowing, normalization, etc.
        if do_preprocessing:
            print("Preprocessing data...")
            train_main, train_appliances = preprocess(
                sequence_length=self.sequence_length,
                mains_std=self.mains_std,
                mains_mean=self.mains_mean,
                mains_lst=train_main,
                submeters_lst=train_appliances,
                method="train",
                appliance_params=self.appliance_params,
                windowing=False
            )
        
        # Prepare main power data for training
        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        
        # Prepare appliance power data
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        
        print(f"Training data shape: {train_main.shape}")
        
        # Train a separate model for each appliance
        appliance_progress = tqdm(train_appliances, desc="Training appliances", unit="appliance")
        
        for appliance_name, power in appliance_progress:
            appliance_progress.set_postfix({"Current": appliance_name})
            
            # Create new model if this appliance hasn't been seen before
            if appliance_name not in self.models:
                print(f"\nFirst model training for {appliance_name}")
                self.models[appliance_name] = self.return_network()
            else:
                print(f"\nStarted Retraining model for {appliance_name}")
            
            model = self.models[appliance_name]
            
            # Train only if we have sufficient data
            if train_main.size > 0:
                if len(train_main) > 10:
                    # Convert to PyTorch tensors and move to device
                    train_x = torch.FloatTensor(train_main).to(self.device)
                    train_y = torch.FloatTensor(power).to(self.device)
                    
                    # Split data into training and validation sets
                    train_x_split, val_x_split, train_y_split, val_y_split = train_test_split(
                        train_x.cpu().numpy(), train_y.cpu().numpy(), 
                        test_size=0.15, random_state=42
                    )
                    
                    # Convert back to tensors and move to device
                    train_x_split = torch.FloatTensor(train_x_split).to(self.device)
                    val_x_split = torch.FloatTensor(val_x_split).to(self.device)
                    train_y_split = torch.FloatTensor(train_y_split).to(self.device)
                    val_y_split = torch.FloatTensor(val_y_split).to(self.device)
                    
                    # Create PyTorch DataLoaders for batch processing
                    train_dataset = TensorDataset(train_x_split, train_y_split)
                    val_dataset = TensorDataset(val_x_split, val_y_split)
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                    
                    # Train the model
                    self.train_model(model, train_loader, val_loader, appliance_name, current_epoch)
    
    def train_model(self, model, train_loader, val_loader, appliance_name, current_epoch):
        """Train a single appliance model with early stopping based on validation loss"""
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        epoch_progress = tqdm(range(self.n_epochs), desc=f"Training {appliance_name}", unit="epoch")
        
        for epoch in epoch_progress:
            # Training phase
            model.train()
            train_loss = 0.0
            
            train_batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", 
                                      leave=False, unit="batch")
            
            for batch_x, batch_y in train_batch_progress:
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batch_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            val_batch_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", 
                                    leave=False, unit="batch")
            
            with torch.no_grad():
                for batch_x, batch_y in val_batch_progress:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    val_loss += loss.item()
                    val_batch_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # Calculate average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            epoch_progress.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best": f"{best_val_loss:.4f}"
            })
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epoch_progress.write(f'New best model saved with val_loss: {val_loss:.4f}')
                
                # Save model checkpoint
                filepath = f"{self.file_prefix}-{appliance_name.replace(' ', '_')}-epoch{current_epoch}.pth"
                torch.save(best_model_state, filepath)
        
        # Load the best model weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nLoaded best model for {appliance_name} with validation loss: {best_val_loss:.4f}")
    
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """Disaggregate power consumption for each appliance from aggregate mains data"""
        
        if model is not None:
            self.models = model
        
        # Preprocess test data similar to training data
        if do_preprocessing:
            print("Preprocessing test data...")
            test_main_list = preprocess(
                sequence_length=self.sequence_length,
                mains_lst=test_main_list,
                mains_mean=self.mains_mean,
                mains_std=self.mains_std,
                submeters_lst=None,
                method="test",
                appliance_params=self.appliance_params,
                windowing=False
            )
        
        test_predictions = []
        
        chunk_progress = tqdm(test_main_list, desc="Processing test chunks", unit="chunk")
        
        # Process each chunk of test data
        for test_main in chunk_progress:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            test_main_tensor = torch.FloatTensor(test_main).to(self.device)
            
            disggregation_dict = {}
            
            appliance_progress = tqdm(self.models.items(), desc="Disaggregating appliances", 
                                    leave=False, unit="appliance")
            
            # Get predictions from each appliance model
            for appliance, model in appliance_progress:
                appliance_progress.set_postfix({"Current": appliance})
                
                model.eval()
                
                # Create DataLoader for batched inference
                test_dataset = TensorDataset(test_main_tensor)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
                predictions = []
                
                pred_progress = tqdm(test_loader, desc=f"Predicting {appliance}", 
                                   leave=False, unit="batch")
                
                # Generate predictions
                with torch.no_grad():
                    for batch_x, in pred_progress:
                        batch_pred = model(batch_x)
                        predictions.append(batch_pred.cpu().numpy())
                
                prediction = np.concatenate(predictions, axis=0)
                
                # Denormalize predictions back to original power scale
                prediction = (self.appliance_params[appliance]['mean'] + 
                            prediction * self.appliance_params[appliance]['std'])
                
                # Ensure non-negative power values
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            
            # Combine all appliance predictions for this chunk
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        
        return test_predictions
    
    def return_network(self):
        """Factory method to create a new RNN model instance"""
        model = RNNModel(self.sequence_length).to(self.device)
        return model
    

    def set_appliance_params(self, train_appliances):
        """Compute normalization statistics (mean, std) for each appliance"""
        print("Setting appliance parameters...")
        
        param_progress = tqdm(train_appliances, desc="Computing appliance stats", unit="appliance")
        
        for (app_name, df_list) in param_progress:
            param_progress.set_postfix({"Current": app_name})
            
            # Concatenate all data for this appliance and compute statistics
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            
            # Prevent division by zero in normalization
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
        
        print(self.appliance_params)