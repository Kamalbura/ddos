from typing import Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

"""## TST"""

class _TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_k if d_k is not None else d_model // n_heads
        d_v = d_v if d_v is not None else d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_act_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src, prev=None, key_padding_mask=None, attn_mask=None):

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class _TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class _TSTBackbone(nn.Module):
    def __init__(self, c_in, seq_len, max_seq_len=512,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len: # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(Pad1d(padding), nn.Conv1d(c_in, d_model, kernel_size=tr_factor, padding=0, stride=tr_factor))
            if verbose: print(f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n')
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs) # Eq 2
            if verbose: print(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n')
        else:
            self.W_P = nn.Linear(c_in, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = self._positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        self.transpose = Transpose(-1, -2, contiguous=True)
        self.key_padding_mask, self.padding_var, self.attn_mask = key_padding_mask, padding_var, attn_mask

    def forward(self, inp):
        # x and padding mask
        if isinstance(inp, tuple): x, key_padding_mask = inp
        elif self.key_padding_mask == 'auto': x, key_padding_mask = self._key_padding_mask(inp) # automatically identify padding mask
        elif self.key_padding_mask == -1: x, key_padding_mask = inp[:, :-1], inp[:, -1]         # padding mask is the last channel
        else: x, key_padding_mask = inp, None

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1))              # Eq 1        # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u, key_padding_mask=key_padding_mask, attn_mask=self.attn_mask)    # z: [bs x q_len x d_model]
        z = self.transpose(z)                                                               # z: [bs x d_model x q_len]
        if key_padding_mask is not None:
            z = z * torch.logical_not(key_padding_mask.unsqueeze(1))  # zero-out padding embeddings
        return z

    def _positional_encoding(self, pe, learn_pe, q_len, d_model):
        # Positional encoding
        if pe == None:
            W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False
        elif pe == 'zero':
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe == 'zeros':
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe == 'normal' or pe == 'gauss':
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pe == 'uniform':
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
        elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
        elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        return nn.Parameter(W_pos, requires_grad=learn_pe)

    def _key_padding_mask(self, x):
        if self.padding_var is not None:
            mask = TSMaskTensor(x[:, self.padding_var] == 1)            # key_padding_mask: [bs x q_len]
            return x, mask
        else:
            mask = torch.isnan(x)
            x[mask] = 0
            if mask.any():
                mask = TSMaskTensor((mask.float().mean(1)==1).bool())   # key_padding_mask: [bs x q_len]
                return x, mask
            else:
                return x, None

class TSTPlus(nn.Sequential):
    """TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs"""
    def __init__(self, c_in:int, c_out:int, seq_len:int, max_seq_len=512,
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k=None, d_v=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var=None, attn_mask=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, flatten:bool=True, fc_dropout:float=0.,
                 concat_pool:bool=False, bn:bool=False, custom_head=None,
                 y_range=None, verbose:bool=False, **kwargs):
        
        # Backbone
        backbone = _TSTBackbone(c_in, seq_len=seq_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model
        self.c_out = c_out
        self.seq_len = backbone.seq_len
        if custom_head is not None:
            if isinstance(custom_head, nn.Module): head = custom_head
            else: head = custom_head(self.head_nf, c_out, seq_len)
        else: head = self.create_head(self.head_nf, c_out, self.seq_len, act=act, flatten=flatten, concat_pool=concat_pool,
                                           fc_dropout=fc_dropout, bn=bn, y_range=y_range)
        super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))

    def create_head(self, nf, c_out, seq_len, flatten=True, concat_pool=False, act="gelu", fc_dropout=0., bn=False, y_range=None):
        layers = [get_act_fn(act)]
        if flatten:
            nf *= seq_len
            layers += [Flatten()]
        else:
            if concat_pool: nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out, bn=bn, p=fc_dropout)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

# Support classes and functions
def ifnone(a, b):
    """Return if a is None, otherwise a."""
    return b if a is None else a

def get_act_fn(act):
    if isinstance(act, nn.Module): return act
    if act is None: return act
    act = act.lower()
    if act == 'relu': return nn.ReLU()
    elif act == 'leakyrelu': return nn.LeakyReLU()
    elif act == 'elu': return nn.ELU()
    elif act == 'gelu': return nn.GELU()
    elif act == 'tanh': return nn.Tanh()
    elif act == 'sigmoid': return nn.Sigmoid()
    elif act == 'softmax': return nn.Softmax(dim=-1)
    else: raise ValueError(f"act_fn {act} not in [None, 'relu', 'leakyrelu', 'tanh', 'sigmoid', 'softmax', nn.Module()]")

class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        layers = lin+layers if lin_first else layers+lin
        if act is not None: layers.append(act)
        super().__init__(*layers)

class GACP1d(nn.Module):
    "Global Adaptive Concat Pooling for 1D inputs"
    def __init__(self, output_size=1): self.output_size = output_size
    def forward(self, x): 
        if self.output_size==1: return torch.cat([F.adaptive_max_pool1d(x, 1), F.adaptive_avg_pool1d(x, 1)], 1)
        return torch.cat([F.adaptive_max_pool1d(x, self.output_size),
                         F.adaptive_avg_pool1d(x, self.output_size)], 1)

class GAP1d(nn.Module):
    "Global Adaptive Pooling for 1D inputs"
    def __init__(self, output_size=1): self.output_size = output_size
    def forward(self, x): return F.adaptive_avg_pool1d(x, self.output_size)

class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), -1) if x.dim() > 2 else x

class Pad1d(nn.Module):
    def __init__(self, padding, value=0):
        super().__init__()
        self.padding, self.value = padding, value
        
    def forward(self, x):
        return F.pad(x, self.padding, value=self.value)
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

class SigmoidRange(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        assert min_val < max_val
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x): return torch.sigmoid(x) * (self.max_val - self.min_val) + self.min_val

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    x = torch.arange(0, q_len).float().view(-1, 1) / q_len
    if exponential: x = torch.exp(torch.tensor([1000]) ** (-x))
    if normalize: x = 2 * x - 1
    return x

def Coord2dPosEncoding(q_len, d_model=None, exponential=False, normalize=True):
    if d_model is None: d_model = q_len
    feat = torch.arange(0, d_model, 2).float() / d_model
    if exponential: feat = 1000 ** (-feat)
    x = torch.arange(0, q_len).float().view(-1, 1) / q_len
    if normalize: x = 2 * x - 1
    PE = torch.zeros(q_len, d_model)
    PE[:, 0::2] = torch.sin(x * feat.view(1, -1).repeat(q_len, 1))
    PE[:, 1::2] = torch.cos(x * feat.view(1, -1).repeat(q_len, 1))
    if d_model % 2 != 0: PE = PE[:, :-1]
    return PE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=5000, normalize=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        self.normalize = normalize
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        if normalize:
            div_term = (position * div_term) / self.max_len
            pe[:, 0::2] = torch.sin(div_term)
            pe[:, 1::2] = torch.cos(div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        if d_model % 2 != 0: pe = pe[:, :-1]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        if self.normalize:
            x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled dot-product attention
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None and self.res_attention: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -1e32)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # logic key_padding_mask with shape [bs x q_len] or [bs x 1 x 1 x q_len]
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e32)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class TSMaskTensor(torch.Tensor): pass