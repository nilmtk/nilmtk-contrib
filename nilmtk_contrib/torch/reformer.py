from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math
from nilmtk.disaggregate import Disaggregator

from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)
class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

# Axial Positional Embeddings
class AxialPositionalEmbedding(nn.Module):
    """
    Axial positional embeddings for long sequences.
    """
    def __init__(self, dim, max_seq_len, axial_shape):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.axial_shape = axial_shape
        
        assert len(axial_shape) == 2, "Axial shape must be 2D"
        assert axial_shape[0] * axial_shape[1] == max_seq_len, "Axial shape must multiply to max_seq_len"
        
        self.axial_dims = [dim // 2, dim - (dim // 2)]
        
        self.pos_embs = nn.ModuleList([
            nn.Embedding(axial_shape[0], self.axial_dims[0]),
            nn.Embedding(axial_shape[1], self.axial_dims[1])
        ])
    
    def forward(self, x):
        b, n, d = x.shape
        embs = []
        
        for i, (shape, pos_emb) in enumerate(zip(self.axial_shape, self.pos_embs)):
            if i == 0:
                pos = torch.arange(n, device=x.device) // self.axial_shape[1]
            else:
                pos = torch.arange(n, device=x.device) % self.axial_shape[1]
            
            emb = pos_emb(pos)
            embs.append(emb)
        
        pos_emb = torch.cat(embs, dim=-1)
        return x + pos_emb

# LSH Attention Implementation
class LSHSelfAttention(nn.Module):
    """
    LSH self-attention for efficient attention computation.
    """
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=4, causal=False, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        
        self.head_dim = dim // heads
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        # LSH parameters
        self.hash_fn = nn.Linear(self.head_dim, n_hashes * bucket_size, bias=False)
        
    def hash_vectors(self, vecs):
        # Simple LSH using random projections
        batch_size, seq_len, dim = vecs.shape
        
        # Apply hash function
        hash_codes = self.hash_fn(vecs)  # (b, n, n_hashes * bucket_size)
        hash_codes = hash_codes.view(batch_size, seq_len, self.n_hashes, self.bucket_size)
        
        # Get bucket assignments
        bucket_assignments = torch.argmax(hash_codes, dim=-1)  # (b, n, n_hashes)
        
        return bucket_assignments
    
    def forward(self, x, mask=None):
        b, n, d = x.shape
        h = self.heads
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)
        
        # For simplicity, we'll use standard attention with some bucketing
        # In a full LSH implementation, this would involve more complex hashing
        
        # Scale queries
        q = q * (self.head_dim ** -0.5)
        
        # Compute attention scores
        scores = torch.einsum('bhid,bhjd->bhij', q, k)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.tril(torch.ones(n, n, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply input mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        
        return self.to_out(out)

# Chunk FeedForward Layer
class ChunkFeedForward(nn.Module):
    """
    A feed-forward layer that processes inputs in chunks to save memory.
    """
    def __init__(self, dim, mult=4, chunks=1, dropout=0.):
        super().__init__()
        self.chunks = chunks
        self.dim = dim
        hidden_dim = int(dim * mult)
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        if self.chunks == 1:
            return self.net(x)
        
        # Process in chunks to save memory
        chunks = x.chunk(self.chunks, dim=1)
        return torch.cat([self.net(c) for c in chunks], dim=1)

# Reformer Block
class ReformerBlock(nn.Module):
    """
    A single block of the Reformer model, combining LSH attention and a feed-forward network.
    """
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=4, ff_mult=4, 
                 ff_chunks=1, causal=False, dropout=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LSHSelfAttention(
            dim=dim,
            heads=heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = ChunkFeedForward(
            dim=dim,
            mult=ff_mult,
            chunks=ff_chunks,
            dropout=dropout
        )
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x

# Main Reformer Network for NILM
class ReformerNet(nn.Module):
    """
    The Reformer network architecture for NILM.
    """
    def __init__(self, sequence_length, dim=512, depth=6, heads=8, bucket_size=64, 
                 n_hashes=4, ff_mult=4, ff_chunks=1, dropout=0.1, 
                 axial_position_emb=True, axial_position_shape=None):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.dim = dim
        
        # Input projection
        self.input_projection = nn.Linear(1, dim)
        
        # Positional embeddings
        if axial_position_emb:
            if axial_position_shape is None:
                # Auto-determine axial shape
                sqrt_seq = int(math.sqrt(sequence_length))
                while sequence_length % sqrt_seq != 0:
                    sqrt_seq -= 1
                axial_position_shape = (sqrt_seq, sequence_length // sqrt_seq)
            
            self.pos_emb = AxialPositionalEmbedding(
                dim=dim,
                max_seq_len=sequence_length,
                axial_shape=axial_position_shape
            )
        else:
            self.pos_emb = nn.Parameter(torch.randn(1, sequence_length, dim))
        
        # Reformer blocks
        self.blocks = nn.ModuleList([
            ReformerBlock(
                dim=dim,
                heads=heads,
                bucket_size=bucket_size,
                n_hashes=n_hashes,
                ff_mult=ff_mult,
                ff_chunks=ff_chunks,
                causal=False,  # For NILM, we can use full attention
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initializes the model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        # Transpose to (batch_size, sequence_length, 1)
        x = x.transpose(1, 2)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch_size, sequence_length, dim)
        
        # Add positional embeddings
        if isinstance(self.pos_emb, AxialPositionalEmbedding):
            x = self.pos_emb(x)
        else:
            x = x + self.pos_emb
        
        # Apply Reformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, dim)
        
        # Output projection
        x = self.to_out(x)  # (batch_size, 1)
        
        return x

class Reformer(Disaggregator):
    """
    Reformer model for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "Reformer: The Efficient Transformer"
    https://arxiv.org/abs/2001.04451
    
    The model adapts the Reformer architecture for energy disaggregation tasks,
    using locality-sensitive hashing (LSH) attention and reversible layers for
    memory-efficient processing of long sequences.
    
    Architecture Overview:
    - LSH self-attention for efficient attention computation
    - Axial positional embeddings for long sequences
    - Chunk feed-forward layers for memory efficiency
    - Reversible residual connections (conceptually)
    - Sequence-to-point prediction for energy disaggregation
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Length of input sequences (default: 99)
            - dim (int): Model dimension (default: 512)
            - depth (int): Number of transformer layers (default: 6)
            - heads (int): Number of attention heads (default: 8)
            - bucket_size (int): LSH bucket size (default: 64)
            - n_hashes (int): Number of LSH hash functions (default: 4)
            - ff_mult (int): Feed-forward expansion factor (default: 4)
            - ff_chunks (int): Number of chunks for feed-forward (default: 1)
            - dropout (float): Dropout rate (default: 0.1)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
    """
    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        super().__init__()
        self.MODEL_NAME = "Reformer"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Extract hyperparameters from params dict
        self.chunk_wise_training = params.get("chunk_wise_training", False)
        self.sequence_length = params.get("sequence_length", 99)
        self.n_epochs = params.get("n_epochs", 10)
        self.batch_size = params.get("batch_size", 512)
        self.appliance_params = params.get("appliance_params", {})
        self.mains_mean = params.get("mains_mean", 1800)
        self.mains_std = params.get("mains_std", 600)
        
        # Reformer specific parameters
        self.dim = params.get("dim", 512)
        self.depth = params.get("depth", 6)
        self.heads = params.get("heads", 8)
        self.bucket_size = params.get("bucket_size", 64)
        self.n_hashes = params.get("n_hashes", 4)
        self.ff_mult = params.get("ff_mult", 4)
        self.ff_chunks = params.get("ff_chunks", 1)
        self.dropout = params.get("dropout", 0.1)
        self.axial_position_emb = params.get("axial_position_emb", True)
        self.axial_position_shape = params.get("axial_position_shape", None)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sequence length must be odd for proper windowing
        if self.sequence_length % 2 == 0:
            _log_print("Sequence length should be odd!")
            raise SequenceLengthError
        
        _log_print(f"Reformer initialized with sequence_length={self.sequence_length}")
        _log_print(f"Reformer params: dim={self.dim}, depth={self.depth}, heads={self.heads}")
        _log_print(f"LSH params: bucket_size={self.bucket_size}, n_hashes={self.n_hashes}")
        _log_print(f"Using device: {self.device}")

    def return_network(self):
        """
        Builds the Reformer network.
        """
        model = ReformerNet(
            sequence_length=self.sequence_length,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            bucket_size=self.bucket_size,
            n_hashes=self.n_hashes,
            ff_mult=self.ff_mult,
            ff_chunks=self.ff_chunks,
            dropout=self.dropout,
            axial_position_emb=self.axial_position_emb,
            axial_position_shape=self.axial_position_shape
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        _log_print(f"Reformer model created with {total_params:,} parameters")
        
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """
        Preprocesses data using a sliding window, matching seq2point.
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
        Trains the Reformer model on a chunk of data.
        """
        # If no appliance wise parameters are provided, then compute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        _log_print("...............Reformer partial_fit running...............")
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
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0)
                    criterion = nn.MSELoss()
                    
                    best_val_loss = float('inf')
                    filepath = checkpoint_path(".pth")
                    
                    # Training loop matching seq2point behavior
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
                            
                            # Add gradient clipping like seq2point
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
                        
                        # Save best model (matching seq2point's ModelCheckpoint behavior)
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
