import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from warnings import warn
from nilmtk.disaggregate import Disaggregator
from tqdm import tqdm  # Added for progress bars

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        
    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, maxlen):
        super(PositionalEncoding, self).__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, maxlen, embed_dim))

    def forward(self, x):
        return x + self.pos_emb

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
        self.maxlen = maxlen
        
    def forward(self, x):
        positions = torch.arange(0, self.maxlen, dtype=torch.long, device=x.device)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class LPpool(nn.Module):
    def __init__(self, pool_size, stride=None, padding=0):
        super(LPpool, self).__init__()
        self.avgpool = nn.AvgPool1d(pool_size, stride=stride, padding=padding)
        
    def forward(self, x):
        x = torch.pow(torch.abs(x), 2)
        x = self.avgpool(x)
        x = torch.pow(x, 1.0 / 2)
        return x

class NILMDataset(Dataset):
    def __init__(self, mains, appliances, sequence_length):
        self.mains = mains
        self.appliances = appliances
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.mains)
    
    def __getitem__(self, idx):
        return self.mains[idx], self.appliances[idx]

class BERT(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "BERT"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def return_network(self):
        embed_dim = 32
        num_heads = 2
        ff_dim = 32
        vocab_size = 20000
        maxlen = self.sequence_length
        
        model = nn.Sequential(
            Permute(0, 2, 1),  # [B, 1, 99]
            nn.Conv1d(1, embed_dim, 4, stride=1, padding='same'),  # [B, embed_dim, 99]
            LPpool(pool_size=2),  # [B, embed_dim, 49]
            Permute(0, 2, 1),  # [B, 49, embed_dim]
            PositionalEncoding(embed_dim, 49),  # [B, 49, embed_dim]
            TransformerBlock(embed_dim, num_heads, ff_dim),  # [B, 49, embed_dim]
            nn.Flatten(),  # [B, 49 * embed_dim]
            nn.Dropout(0.1),
            nn.Linear(49 * embed_dim, self.sequence_length),
            nn.Dropout(0.1)
        ).to(self.device)
        
        return model
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        print("...............BERT partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
            
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
                
        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        
        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances
        
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)
                
            model = self.models[appliance_name]
            optimizer = optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            
            if train_main.size > 0:
                if len(train_main) > 10:
                    train_x, v_x, train_y, v_y = train_test_split(
                        train_main, power, test_size=.15, random_state=10)
                    
                    train_dataset = NILMDataset(train_x, train_y, self.sequence_length)
                    val_dataset = NILMDataset(v_x, v_y, self.sequence_length)
                    
                    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
                    
                    best_val_loss = float('inf')
                    
                    for epoch in range(self.n_epochs):
                        # Training phase with tqdm
                        model.train()
                        train_loss = 0.0
                        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs} [Train]", leave=False)
                        for batch_mains, batch_appliances in train_loop:
                            batch_mains = batch_mains.float().to(self.device)
                            batch_appliances = batch_appliances.float().to(self.device)
                            
                            optimizer.zero_grad()
                            outputs = model(batch_mains)
                            loss = criterion(outputs, batch_appliances)
                            loss.backward()
                            optimizer.step()
                            
                            train_loss += loss.item() * batch_mains.size(0)
                            train_loop.set_postfix(loss=loss.item())
                        
                        train_loss /= len(train_loader.dataset)
                        
                        # Validation phase with tqdm
                        model.eval()
                        val_loss = 0.0
                        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.n_epochs} [Val]", leave=False)
                        with torch.no_grad():
                            for batch_mains, batch_appliances in val_loop:
                                batch_mains = batch_mains.float().to(self.device)
                                batch_appliances = batch_appliances.float().to(self.device)
                                
                                outputs = model(batch_mains)
                                loss = criterion(outputs, batch_appliances)
                                val_loss += loss.item() * batch_mains.size(0)
                                val_loop.set_postfix(loss=loss.item())
                            
                            val_loss /= len(val_loader.dataset)
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                torch.save(model.state_dict(), f'BERT-temp-weights-{appliance_name}.pt')
                        
                        print(f'Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    
                    model.load_state_dict(torch.load(f'BERT-temp-weights-{appliance_name}.pt'))

    # [Rest of the methods remain exactly the same as in the previous version]
    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model
            
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')
                
        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))
            
            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                
                test_dataset = NILMDataset(test_main_array, np.zeros((len(test_main_array), self.sequence_length), dtype=np.float32),self.sequence_length)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                
                prediction = []
                with torch.no_grad():
                    # Add tqdm for prediction progress
                    for batch_mains, _ in tqdm(test_loader, desc=f"Disaggregating {appliance}", leave=False):
                        batch_mains = batch_mains.float().to(self.device)
                        outputs = model(batch_mains)
                        prediction.append(outputs.cpu().numpy())
                
                prediction = np.concatenate(prediction, axis=0)
                
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]
                
                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
                
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
            
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
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
                else:
                    print("Parameters for ", app_name, " were not found!")
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
    
    def set_appliance_params(self, train_appliances):
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})