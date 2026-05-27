"""
NILMFormer: PyTorch Implementation for NILMTK-Contrib

This is a NILMFormer-inspired implementation based on the paper:
"NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity"
by Petralia et al. (ACM SIGKDD 2025)

Official GitHub: https://github.com/adrienpetralia/NILMFormer
Paper: https://arxiv.org/html/2506.05880v1

Architecture components to audit against the official implementation:
1. Instance Normalization: Stationarizes input by subtracting mean/std
2. DilatedBlock: Robust convolutional feature extractor with residual connections
3. TokenStats: Linear projection of mean/std statistics into higher dimensional space
4. Exogenous Features: Temporal encoding using create_exogene (sinusoidal functions for
   month, day-of-week, hour, minute)
5. Transformer Encoder: Diagonal masked self-attention with pre-norm architecture
6. Output Head: 1D convolution for sequence-to-sequence prediction
7. Denormalization: Reverse instance normalization using projected statistics

Key Features:
- create_exogene for capturing temporal patterns (from original NILMFormer repo)
- Diagonal masking (not causal) in self-attention
- GELU activations throughout
- Pre-norm transformer blocks
- Instance normalization for non-stationarity handling
- Sequence-to-sequence prediction with middle-point extraction
- Parameter defaults intended to track the official config (d_model=96, n_heads=8, etc.)

This implementation adapts NILMFormer concepts to the NILMTK-Contrib
Disaggregator interface. Source parity must be verified before making
reproduction claims.
"""

from typing import List, Optional
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nilmtk.disaggregate import Disaggregator
from nilmtk_contrib.utils.model import initialize_runtime, legacy_print, module_logger, checkpoint_path

logger = module_logger(__name__)
_log_print = legacy_print(logger)


class SequenceLengthError(Exception):
    pass


class ApplianceNotFoundError(Exception):
    pass


class NILMDataset(Dataset):
    """
    Dataset class for NILMFormer.
    """
    def __init__(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Input tensor of shape (B, C, L), where C includes
                             mains power and exogenous features.
            targets (Tensor): Target tensor of shape (B, C_out, L), where C_out
                              is the number of appliances.
        """
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ResUnit(nn.Module):
    """
    Residual Unit for the NILMFormer model.
    """
    def __init__(self, c_in: int, c_out: int, k: int = 8, dilation: int = 1, 
                 stride: int = 1, bias: bool = True):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=k,
                dilation=dilation,
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x) -> torch.Tensor:
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    """
    Dilated Convolutional Block for feature extraction.
    """
    def __init__(self, c_in: int = 1, c_out: int = 72, kernel_size: int = 8,
                 dilation_list: Optional[List[int]] = None, bias: bool = True):
        super().__init__()
        
        if dilation_list is None:
            dilation_list = [1, 2, 4, 8]

        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(
                    ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
            else:
                layers.append(
                    ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


def create_exogene(start_date, sequence_length, freq="1min", 
                   list_exo_variables=None, cosinbase=True, new_range=(-1, 1)):
    """
    Creates exogenous temporal features.
    
    Args:
        start_date: The starting timestamp for the sequence.
        sequence_length: The length of the time sequence.
        freq: The frequency of the data sampling.
        list_exo_variables: A list of temporal features to generate.
        cosinbase: If True, uses sinusoidal encoding for features.
        new_range: The range for normalization if cosinbase is False.
    
    Returns:
        An array of exogenous features.
    """
    if list_exo_variables is None:
        list_exo_variables = ['month', 'dow', 'hour', 'minute']  # Default temporal features
    
    if cosinbase:
        n_var = 2 * len(list_exo_variables)  # sin and cos for each variable
    else:
        n_var = len(list_exo_variables)
    
    # Create datetime range
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    tmp = pd.date_range(start=start_date, periods=sequence_length, freq=freq)
    
    # Initialize exogenous features array
    np_extra = np.zeros((1, n_var, sequence_length)).astype(np.float32)
    
    k = 0
    for exo_var in list_exo_variables:
        if exo_var == "month":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.month.values / 12.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.month.values / 12.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=12, newRange=new_range
                )
                k += 1
        elif exo_var == "dom":  # day of month
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.day.values / 31.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.day.values / 31.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.day.values, xmin=1, xmax=31, newRange=new_range
                )
                k += 1
        elif exo_var == "dow":  # day of week
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.dayofweek.values / 7.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.dayofweek.values / 7.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.dayofweek.values, xmin=0, xmax=6, newRange=new_range
                )
                k += 1
        elif exo_var == "hour":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.hour.values / 24.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.hour.values / 24.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.hour.values, xmin=0, xmax=23, newRange=new_range
                )
                k += 1
        elif exo_var == "minute":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.minute.values / 60.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.minute.values / 60.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.minute.values, xmin=0, xmax=59, newRange=new_range
                )
                k += 1
        else:
            raise ValueError(
                f"Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour', 'minute' supported, received {exo_var}"
            )
    
    return np_extra


def normalize_exogene(x, xmin, xmax, newRange):
    """
    Normalizes exogenous features to a specified range.
    """
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    
    norm = (x - xmin) / (xmax - xmin)
    if newRange == (0, 1):
        return norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]


class DiagonalMaskFromSeqlen:
    """
    Creates a diagonal attention mask.
    """
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            self._mask = torch.diag(
                torch.ones(L, dtype=torch.bool, device=device)
            ).repeat(B, 1, 1, 1)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class DiagonallyMaskedSelfAttention(nn.Module):
    """
    Self-attention mechanism with a diagonal mask.
    """
    def __init__(self, dim: int, n_heads: int, head_dim: int, dropout: float):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.dropout: float = dropout
        self.scale = head_dim**-0.5

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(batch, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(batch, seqlen, self.n_heads, self.head_dim)

        diag_mask = DiagonalMaskFromSeqlen(batch, seqlen, device=xq.device)

        scale = 1.0 / xq.shape[-1] ** 0.5
        scores = torch.einsum("blhe,bshe->bhls", xq, xk)
        attn = self.attn_dropout(
            torch.softmax(
                scale * scores.masked_fill_(diag_mask.mask, -np.inf), dim=-1
            )
        )
        output = torch.einsum("bhls,bshd->blhd", attn, xv)

        return self.out_dropout(self.wo(output.reshape(batch, seqlen, -1)))


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, dim: int, hidden_dim: int, dp_rate: float = 0.0, 
                 bias1: bool = True, bias2: bool = True):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = F.gelu

    def forward(self, x) -> torch.Tensor:
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with pre-norm architecture.
    """
    def __init__(self, d_model: int, n_heads: int, dp_rate: float = 0.2, 
                 pffn_ratio: int = 4, norm_eps: float = 1e-5):
        super().__init__()
        
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.attention_layer = DiagonallyMaskedSelfAttention(
            dim=d_model,
            n_heads=n_heads,
            head_dim=d_model // n_heads,
            dropout=dp_rate,
        )

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout = nn.Dropout(dp_rate)

        self.pffn = PositionWiseFeedForward(
            dim=d_model,
            hidden_dim=d_model * pffn_ratio,
            dp_rate=dp_rate,
        )

    def forward(self, x) -> torch.Tensor:
        # Pre-norm attention block
        x = self.norm1(x)
        new_x = self.attention_layer(x)
        x = torch.add(x, new_x)

        # Pre-norm PFFN block
        x = self.norm2(x)
        new_x = self.pffn(x)
        x = torch.add(x, self.dropout(new_x))

        return x


class NILMFormerNetwork(nn.Module):
    """
    The NILMFormer neural network architecture.
    """
    def __init__(self, c_in=1, c_embedding=8, c_out=1, kernel_size=3, 
                 kernel_size_head=3, dilations=None, conv_bias=True,
                 n_encoder_layers=3, d_model=96, dp_rate=0.2, pffn_ratio=4,
                 n_heads=8, norm_eps=1e-5):
        super().__init__()
        
        if dilations is None:
            dilations = [1, 2, 4, 8]
            
        # Validate constraints
        assert d_model % 4 == 0, "d_model must be divisible by 4."
        
        # Store config
        self.d_model = d_model
        self.c_out = c_out
        
        # ============ Embedding ============#
        d_model_ = 3 * d_model // 4  # e.g., if d_model=96 => d_model_=72

        self.EmbedBlock = DilatedBlock(
            c_in=c_in,
            c_out=d_model_,
            kernel_size=kernel_size,
            dilation_list=dilations,
            bias=conv_bias,
        )

        # Exogenous input projection (from create_exogene features)
        self.ProjEmbedding = nn.Conv1d(
            in_channels=c_embedding, 
            out_channels=d_model // 4, 
            kernel_size=1
        )

        self.ProjStats1 = nn.Linear(2, d_model)
        self.ProjStats2 = nn.Linear(d_model, 2)

        # ============ Encoder ============#
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(d_model, n_heads, dp_rate, pffn_ratio, norm_eps))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = nn.Sequential(*layers)

        # ============ Downstream Task Head ============#
        self.DownstreamTaskHead = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=kernel_size_head,
            padding=kernel_size_head // 2,
            padding_mode="replicate",
        )

        # ============ Initialize Weights ============#
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the linear and layer normalization layers.
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass for the NILMFormer model.
        
        Args:
            x (Tensor): Input tensor of shape (B, 1 + e, L), where B is the batch size,
                        e is the number of exogenous features, and L is the sequence length.
        
        Returns:
            Tensor: The output of the model.
        """
        # Separate the channels:
        #   x[:, :1, :] => load curve
        #   x[:, 1:, :] => exogenous input(s)
        encoding = x[:, 1:, :]  # shape: (B, e, L)
        x = x[:, :1, :]  # shape: (B, 1, L)

        # === Instance Normalization === #
        inst_mean = torch.mean(x, dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6
        ).detach()

        x = (x - inst_mean) / inst_std  # shape still (B, 1, L)

        # === Embedding === #
        # 1) Dilated Conv block
        x = self.EmbedBlock(x)  # shape: (B, [d_model_], L) => typically (B, 72, L) if d_model=96
        
        # 2) Project exogenous features
        encoding = self.ProjEmbedding(encoding)  # shape: (B, d_model//4, L)
        
        # 3) Concatenate dilated features with exogenous features
        x = torch.cat([x, encoding], dim=1).permute(0, 2, 1)  # (B, L, d_model)

        # === Mean/Std tokens === #
        stats_token = self.ProjStats1(
            torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)
        )  # (B, 1, d_model)
        x = torch.cat([x, stats_token], dim=1)  # (B, L + 1, d_model)

        # === Transformer Encoder === #
        x = self.EncoderBlock(x)  # (B, L + 1, d_model)
        x = x[:, :-1, :]  # remove stats token => (B, L, d_model)

        # === Conv Head === #
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.DownstreamTaskHead(x)  # (B, c_out, L)

        # === Reverse Instance Normalization === #
        # stats_out => shape (B, 1, 2)
        stats_out = self.ProjStats2(stats_token)  # stats_token was (B, 1, d_model)
        outinst_mean = stats_out[:, :, 0].unsqueeze(-1)  # (B, 1, 1)
        outinst_std = stats_out[:, :, 1].unsqueeze(-1)  # (B, 1, 1)

        x = x * outinst_std + outinst_mean
        return x


class NILMFormer(Disaggregator):
    """
    NILMFormer: Transformer-based model for non-intrusive load monitoring.
    
    This implementation is based on the paper:
    "NILMFormer: Non-Intrusive Load Monitoring that Accounts for Non-Stationarity"
    https://arxiv.org/abs/2506.05880
    
    The model uses a transformer architecture specifically designed for energy disaggregation 
    tasks that addresses non-stationarity in power consumption data through instance 
    normalization and temporal feature encoding.
    
    Architecture Overview:
    - Instance normalization for handling non-stationarity
    - Dilated convolutional feature extractor with residual connections
    - Exogenous temporal features (month, day-of-week, hour, minute)
    - Transformer encoder with diagonal masked self-attention
    - Sequence-to-sequence prediction with denormalization
    
    Parameters:
        params (dict): Configuration parameters including:
            - sequence_length (int): Input sequence length (default: 99)
            - c_in (int): Input channels (default: 1)
            - c_embedding (int): Exogenous channels (default: 8)
            - d_model (int): Model dimension (default: 96)
            - n_heads (int): Number of attention heads (default: 8)
            - n_layers (int): Number of transformer layers (default: 6)
            - n_epochs (int): Number of training epochs (default: 10)
            - batch_size (int): Training batch size (default: 512)
    """

    def __init__(self, params):
        initialize_runtime(self, params, backends=("python", "numpy", "torch"))
        """
        Initialize NILMFormer model with specified parameters following the paper
        
        Parameters:
        -----------
        params : dict
            Dictionary containing model parameters:
            - sequence_length: Input sequence length (default: 99)
            - c_in: Input channels (default: 1) 
            - c_embedding: Exogenous channels (default: 8)
            - c_out: Output channels (default: 1)
            - d_model: Model dimension (default: 96)
            - n_heads: Number of attention heads (default: 8)
            - n_encoder_layers: Number of encoder layers (default: 3)
            - dp_rate: Dropout rate (default: 0.2)
            - pffn_ratio: Feed-forward expansion ratio (default: 4)
            - kernel_size: Conv kernel size (default: 3)
            - dilations: Dilation factors (default: [1, 2, 4, 8])
            - n_epochs: Training epochs (default: 100)
            - batch_size: Batch size (default: 1024)
            - learning_rate: Learning rate (default: 1e-4)
        """
        super().__init__()
        
        self.MODEL_NAME = "NILMFormer"
        self.models = OrderedDict()
        self.file_prefix = f"{self.MODEL_NAME.lower()}-temp-weights"
        
        # Model architecture parameters intended to follow NILMFormer defaults.
        self.sequence_length = params.get('sequence_length', 99)
        self.c_in = params.get('c_in', 1)
        self.c_embedding = params.get('c_embedding', 8)
        self.c_out = params.get('c_out', 1)
        self.d_model = params.get('d_model', 96)
        self.n_heads = params.get('n_heads', 8)
        self.n_encoder_layers = params.get('n_encoder_layers', 3)
        self.dp_rate = params.get('dp_rate', 0.2)
        self.pffn_ratio = params.get('pffn_ratio', 4)
        self.kernel_size = params.get('kernel_size', 3)
        self.kernel_size_head = params.get('kernel_size_head', 3)
        self.dilations = params.get('dilations', [1, 2, 4, 8])
        self.conv_bias = params.get('conv_bias', True)
        self.norm_eps = params.get('norm_eps', 1e-5)
        
        # Training parameters (optimized for NILMFormer)
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.n_epochs = params.get('n_epochs', 100)  # More epochs for transformer
        self.batch_size = params.get('batch_size', 1024)  # Larger batch size
        self.learning_rate = params.get('learning_rate', 1e-4)  # Lower learning rate
        self.warmup_steps = params.get('warmup_steps', 1000)  # Learning rate warmup
        
        # Data parameters
        self.appliance_params = params.get('appliance_params', {})
        self.mains_mean = params.get('mains_mean', 1800)
        self.mains_std = params.get('mains_std', 600)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log_print(f"NILMFormer using device: {self.device}")
        
        if self.sequence_length % 2 == 0:
            _log_print("Sequence length should be odd!")
            raise SequenceLengthError()

    def return_network(self):
        """Create and return the NILMFormer-inspired network."""
        model = NILMFormerNetwork(
            c_in=self.c_in,
            c_embedding=self.c_embedding,
            c_out=self.c_out,
            kernel_size=self.kernel_size,
            kernel_size_head=self.kernel_size_head,
            dilations=self.dilations,
            conv_bias=self.conv_bias,
            n_encoder_layers=self.n_encoder_layers,
            d_model=self.d_model,
            dp_rate=self.dp_rate,
            pffn_ratio=self.pffn_ratio,
            n_heads=self.n_heads,
            norm_eps=self.norm_eps
        )
        return model.to(self.device)

    def create_exogene_features(self, n_samples, sequence_length, start_date=None):
        """
        Create exogenous temporal features using the NILMFormer approach.
        
        This function generates sinusoidal temporal features from timestamps,
        following the intended NILMFormer timestamp-feature design.
        
        Args:
            n_samples: Number of samples
            sequence_length: Length of each sequence  
            start_date: Starting date (datetime or None for reference date)
        
        Returns:
            exogenous_features: (n_samples, c_embedding, sequence_length) tensor of temporal features
        """
        if start_date is None:
            # Use a reference date (e.g., start of 2023)
            import datetime
            start_date = datetime.datetime(2023, 1, 1)
        
        # Assume data is sampled every minute (can be adjusted based on dataset)
        freq = "1min"
        
        # Temporal variables to include (following original implementation)
        list_exo_variables = ['month', 'dow', 'hour', 'minute']  # Standard set
        
        all_exogenous = []
        for i in range(n_samples):
            # Each sample starts at a different time
            sample_start = start_date + pd.Timedelta(minutes=i * sequence_length)
            
            # Generate exogenous features for this sample
            exo_features = create_exogene(
                start_date=sample_start,
                sequence_length=sequence_length, 
                freq=freq,
                list_exo_variables=list_exo_variables,
                cosinbase=True,  # Use sin/cos encoding
                new_range=(-1, 1)
            )  # Shape: (1, n_features, sequence_length)
            
            all_exogenous.append(exo_features[0])  # Remove the first dimension
        
        # Stack all samples
        exogenous_tensor = np.stack(all_exogenous, axis=0)  # (n_samples, n_features, sequence_length)
        
        return torch.tensor(exogenous_tensor, dtype=torch.float32)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True,
                   current_epoch=0, **load_kwargs):
        """
        Train NILMFormer model on a data chunk
        """
        
        # Compute appliance parameters if not available
        if not self.appliance_params:
            self.set_appliance_params(train_appliances)

        _log_print("...............NILMFormer partial_fit running...............")
        
        # Preprocess data
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        # Prepare main power data
        train_main = pd.concat(train_main, axis=0)
        train_main_values = train_main.values.reshape((-1, self.sequence_length, 1))
        
        # Create exogenous temporal features using create_exogene (much better than random noise!)
        n_samples = train_main_values.shape[0]
        exogenous_features = self.create_exogene_features(n_samples, self.sequence_length)
        
        # Prepare input: concatenate main power with exogenous features
        # Main power: (B, 1, L), Exogenous: (B, c_embedding, L)
        train_main_tensor = torch.tensor(train_main_values.transpose(0, 2, 1), dtype=torch.float32)  # (B, 1, L)
        train_input = torch.cat([train_main_tensor, exogenous_features], dim=1)  # (B, 1 + c_embedding, L)
        
        # Prepare appliance data
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length, 1))
            app_df_tensor = torch.tensor(app_df_values, dtype=torch.float32)
            new_train_appliances.append((app_name, app_df_tensor))
        train_appliances = new_train_appliances

        # Train models for each appliance
        for appliance_name, power_tensor in train_appliances:
            if appliance_name not in self.models:
                _log_print(f"First model training for {appliance_name}")
                self.models[appliance_name] = self.return_network()
            else:
                _log_print(f"Started Retraining model for {appliance_name}")

            model = self.models[appliance_name]
            
            if train_input.size(0) > 10:
                self.train_model(model, train_input, power_tensor, 
                               appliance_name, current_epoch)

    def train_model(self, model, train_input, power_tensor, appliance_name, current_epoch):
        """Train a single appliance model with proper NILMFormer training protocol"""
        
        # Split data
        n_total = train_input.size(0)
        val_split = int(0.15 * n_total)
        
        indices = torch.randperm(n_total)
        train_indices = indices[val_split:]
        val_indices = indices[:val_split]
        
        train_input_split = train_input[train_indices].to(self.device)
        train_power_split = power_tensor[train_indices].to(self.device)
        
        val_input_split = train_input[val_indices].to(self.device)
        val_power_split = power_tensor[val_indices].to(self.device)
        
        # For NILMFormer, we predict the full sequence
        # Target shape: (batch, sequence_length, 1) -> (batch, 1, sequence_length)
        train_power_split = train_power_split.transpose(1, 2)  # (B, 1, L)
        val_power_split = val_power_split.transpose(1, 2)  # (B, 1, L)
        
        # Create datasets and loaders
        train_dataset = NILMDataset(train_input_split, train_power_split)
        val_dataset = NILMDataset(val_input_split, val_power_split)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup optimizer with weight decay (important for transformers)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01,  # Weight decay for regularization
            betas=(0.9, 0.95)   # Optimized betas for transformers
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * self.n_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        best_model_path = checkpoint_path(".pth")
        patience = 10
        patience_counter = 0
        
        _log_print(f"Training {appliance_name} with {total_steps} total steps using integrated exogenous features")
        
        # Training loop
        for epoch in range(self.n_epochs):
            model.train()
            train_losses = []
            
            # Training phase
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for input_batch, power_batch in train_bar:
                input_batch = input_batch.to(self.device)
                power_batch = power_batch.to(self.device)
                
                optimizer.zero_grad()
                # Forward pass without timestamps
                predictions = model(input_batch)  # Shape: (B, c_out, L)
                loss = criterion(predictions, power_batch)
                loss.backward()
                
                # Gradient clipping (important for transformer stability)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
                train_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for input_batch, power_batch in val_loader:
                    input_batch = input_batch.to(self.device)
                    power_batch = power_batch.to(self.device)
                    
                    predictions = model(input_batch)
                    loss = criterion(predictions, power_batch)
                    val_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            _log_print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model and early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                _log_print(f"Saved best model for {appliance_name}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    _log_print(f"Early stopping triggered for {appliance_name}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        _log_print(f"Training completed for {appliance_name}")

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        """
        Disaggregate power consumption for test data using NILMFormer
        """
        
        if model is not None:
            self.models = model

        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            
            # Store original length before any preprocessing
            original_length = len(test_mains_df)
            
            if do_preprocessing:
                # Use the standard preprocessing pipeline
                processed_mains_list = self.call_preprocessing(
                    [test_mains_df], submeters_lst=None, method='test')
                processed_mains_df = processed_mains_list[0]
                
                # Convert preprocessed data to proper format
                test_main_values = processed_mains_df.values  # Already shaped correctly
                test_main_tensor = torch.tensor(
                    test_main_values.reshape((-1, 1, self.sequence_length)), 
                    dtype=torch.float32
                )  # (N, 1, L)
            else:
                # Manual preprocessing if needed
                test_main_values = test_mains_df.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                test_main_values = np.pad(
                    test_main_values, (units_to_pad, units_to_pad),
                    'constant', constant_values=(0, 0)
                )
                test_main_values = np.array([
                    test_main_values[i:i + n] for i in range(len(test_main_values) - n + 1)
                ])
                test_main_values = (test_main_values - self.mains_mean) / self.mains_std
                test_main_tensor = torch.tensor(
                    test_main_values.reshape((-1, 1, self.sequence_length)),
                    dtype=torch.float32
                )
            
            # Create exogenous temporal features for test data
            n_samples = test_main_tensor.shape[0]
            test_exogenous = self.create_exogene_features(n_samples, self.sequence_length)
            
            # Prepare input: concatenate main power with exogenous features
            test_input = torch.cat([test_main_tensor, test_exogenous], dim=1)  # (B, 1 + c_embedding, L)
            test_input_tensor = test_input.to(self.device)

            for appliance in self.models:
                model = self.models[appliance]
                model.eval()
                
                with torch.no_grad():
                    # Process in batches to avoid memory issues
                    predictions = []
                    for i in range(0, len(test_input_tensor), self.batch_size):
                        batch = test_input_tensor[i:i+self.batch_size]
                        pred_batch = model(batch)  # Shape: (B, c_out, L)
                        predictions.append(pred_batch.cpu().numpy())
                    
                    prediction = np.concatenate(predictions, axis=0)  # (N, c_out, L)

                # Extract middle predictions for sequence-to-point conversion
                middle_idx = self.sequence_length // 2
                point_predictions = prediction[:, 0, middle_idx]  # (N,)
                
                # Reconstruct full sequence using correct overlapping window logic
                padding = self.sequence_length // 2
                reconstructed_length = original_length  # Use original length!
                sum_arr = np.zeros(reconstructed_length + 2 * padding)
                counts_arr = np.zeros(reconstructed_length + 2 * padding)
                
                # Place predictions at correct positions
                for i, pred_value in enumerate(point_predictions):
                    target_idx = i + padding  # Account for padding offset
                    if target_idx < len(sum_arr):
                        sum_arr[target_idx] += pred_value
                        counts_arr[target_idx] += 1
                
                # Average overlapping predictions and extract original sequence
                valid_mask = counts_arr > 0
                final_prediction = np.zeros_like(sum_arr)
                final_prediction[valid_mask] = sum_arr[valid_mask] / counts_arr[valid_mask]
                
                # Extract the original sequence (remove padding)
                final_prediction = final_prediction[padding:padding + original_length]
                
                # Denormalize the predictions
                if appliance in self.appliance_params:
                    app_mean = self.appliance_params[appliance]['mean']
                    app_std = self.appliance_params[appliance]['std']
                    final_prediction = final_prediction * app_std + app_mean
                
                # Clip negative values
                final_prediction_clipped = np.where(final_prediction > 0, final_prediction, 0)
                df = pd.Series(final_prediction_clipped)
                disggregation_dict[appliance] = df

            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        """Preprocess data for training or testing"""
        
        if method == 'train':
            # Training preprocessing
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(
                    new_mains, (units_to_pad, units_to_pad),
                    'constant', constant_values=(0, 0)
                )
                new_mains = np.array([
                    new_mains[i:i + n] for i in range(len(new_mains) - n + 1)
                ])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    _log_print(self.appliance_params)
                    _log_print(f"Parameters for {app_name} were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []
                for app_df in app_df_list:
                    new_app_readings = app_df.values.flatten()
                    n = self.sequence_length
                    units_to_pad = n // 2
                    new_app_readings = np.pad(
                        new_app_readings, (units_to_pad, units_to_pad),
                        'constant', constant_values=(0, 0)
                    )
                    new_app_readings = np.array([
                        new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)
                    ])
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                
                appliance_list.append((app_name, processed_appliance_dfs))
            
            return processed_mains_lst, appliance_list

        else:
            # Test preprocessing
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(
                    new_mains, (units_to_pad, units_to_pad),
                    'constant', constant_values=(0, 0)
                )
                new_mains = np.array([
                    new_mains[i:i + n] for i in range(len(new_mains) - n + 1)
                ])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            
            return processed_mains_lst

    def denormalize_output(self, predictions, appliance_name):
        """Denormalize model predictions for a specific appliance"""
        if appliance_name in self.appliance_params:
            app_mean = self.appliance_params[appliance_name]['mean']
            app_std = self.appliance_params[appliance_name]['std']
            return predictions * app_std + app_mean
        else:
            return predictions

    def set_appliance_params(self, train_appliances):
        """Calculate normalization parameters for each appliance"""
        
        for (app_name, df_list) in train_appliances:
            values = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(values)
            app_std = np.std(values)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({
                app_name: {'mean': app_mean, 'std': app_std}
            })
        
        _log_print("Appliance parameters:", self.appliance_params)
