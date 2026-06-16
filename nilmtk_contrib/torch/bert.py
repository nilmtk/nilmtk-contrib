import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from nilmtk_contrib.utils.validation import safe_train_test_split as train_test_split
from nilmtk.disaggregate import Disaggregator
from tqdm import tqdm  # Added for progress bars

from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)  # reorder for Conv1d


class TransformerBlock(nn.Module):
    """
    Transformer encoder block: multi-head self-attention + feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate, batch_first=True)
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
        # x shape: [batch, seq_len, embed_dim] with batch_first=True
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # x comes in as [B, seq_len, 16] from conv layer
        batch_size, seq_len, features = x.shape
        
        # Convert continuous values to discrete tokens for each feature dimension
        # Take the mean across features and discretize
        x_mean = x.mean(dim=-1)  # [B, seq_len]
        
        # Scale and clamp to vocab range
        x_tokens = torch.clamp((x_mean * 1000).long(), 0, self.token_emb.num_embeddings - 1)
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = self.pos_emb(positions)  # [seq_len, embed_dim]
        
        # Get token embeddings
        token_embs = self.token_emb(x_tokens)  # [B, seq_len, embed_dim]
        
        return token_embs + positions.unsqueeze(0)  # [B, seq_len, embed_dim]

class LPpool(nn.Module):
    def __init__(self, pool_size, stride=None, padding=0):
        super(LPpool, self).__init__()
        if stride is None:
            stride = pool_size
        # For 'same' padding equivalent, calculate padding size
        if padding == 'same':
            padding = (pool_size - 1) // 2
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
    """
    BERT-inspired transformer model for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    https://arxiv.org/abs/1810.04805
    
    The model adapts the BERT transformer architecture for energy disaggregation tasks,
    using a sequence-to-sequence approach to predict individual appliance power consumption
    from aggregate household power measurements.
    
    Architecture Overview:
    - 1D Convolutional layer (16 filters, kernel size 4) for feature extraction
    - LP pooling (pool size 2) for dimensionality reduction
    - Token and position embedding layer to convert continuous values to embeddings
    - Single transformer encoder block with multi-head self-attention
    - Dense output layer for sequence prediction
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 99)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
            - chunk_wise_training (bool): Enable chunk-wise training (default: False)
            - appliance_params (dict): Appliance-specific normalization parameters
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
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
            _log_print("Sequence length should be odd!")
            raise SequenceLengthError
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def return_network(self):
        """Create the BERT-inspired module used by this backend.
        
        Key architectural features:
        - Conv1D(16, 4) with 'same' padding and linear activation
        - LPpool with pool_size=2 
        - TokenAndPositionEmbedding applied to 16-dim features -> 32-dim embeddings
        - Single TransformerBlock 
        - Dense layer mapping to sequence_length output
        """
        embed_dim = 32
        num_heads = 2
        ff_dim = 32
        vocab_size = 20000
        maxlen = 49  # After pooling, sequence length becomes 49 (99 -> 49 after pool_size=2)
        
        class BERTModel(nn.Module):
            def __init__(self, embed_dim, num_heads, ff_dim, vocab_size, maxlen, sequence_length, device):
                super(BERTModel, self).__init__()
                self.permute1 = Permute(0, 2, 1)
                self.conv1d = nn.Conv1d(1, 16, 4, stride=1, padding='same')
                self.lppool = LPpool(pool_size=2)
                self.permute2 = Permute(0, 2, 1)
                self.token_pos_emb = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
                self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
                self.flatten = nn.Flatten()
                self.dropout1 = nn.Dropout(0.1)
                self.linear = nn.Linear(maxlen * embed_dim, sequence_length)  # Use maxlen instead of hardcoded 49
                self.dropout2 = nn.Dropout(0.1)
                
            def forward(self, x):
                x = self.permute1(x)  # [B, 1, 99]
                x = self.conv1d(x)    # [B, 16, 99]
                x = self.lppool(x)    # [B, 16, 49]
                x = self.permute2(x)  # [B, 49, 16]
                x = self.token_pos_emb(x)  # [B, 49, 32]
                x = self.transformer(x)    # [B, 49, 32]
                x = self.flatten(x)        # [B, 49 * 32]
                x = self.dropout1(x)
                x = self.linear(x)         # [B, sequence_length]
                x = self.dropout2(x)
                return x
        
        model = BERTModel(embed_dim, num_heads, ff_dim, vocab_size, maxlen, self.sequence_length, self.device).to(self.device)
        return model
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        _log_print("...............BERT partial_fit running...............")
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
                _log_print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                _log_print("Started Retraining model for ", appliance_name)
                
            model = self.models[appliance_name]
            # Use default Adam parameters to match TF's 'adam'
            optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)
            criterion = nn.MSELoss()
            
            if train_main.size > 0:
                if len(train_main) > 10:
                    # Create unique filename for model weights like TF version
                    filepath = checkpoint_path(".pt")
                    
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
                        
                        train_loss /= len(train_dataset)  # Use dataset length directly
                        
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
                            
                            val_loss /= len(val_dataset)  # Use dataset length directly
                            
                            # Save best model (like ModelCheckpoint in TF)
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                torch.save(model.state_dict(), filepath)
                                _log_print(f'Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - Model saved')
                            else:
                                _log_print(f'Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    
                    # Load best weights (like TF version)
                    model.load_state_dict(torch.load(filepath))

    # Remaining methods keep the legacy backend behavior.
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
                
                window_length = self.sequence_length
                n = len(prediction) + window_length - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                len(sum_arr)
                
                for i in range(len(prediction)):
                    sum_arr[i:i + window_length] += prediction[i].flatten()
                    counts_arr[i:i + window_length] += 1
                
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
                    _log_print("Parameters for ", app_name, " were not found!")
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
                # TF version doesn't pad during test - comment out padding line
                # new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst
    
    def set_appliance_params(self, train_appliances):
        for (app_name, df_list) in train_appliances:
            values = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
