from torch import nn
import torch
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attn(self, Q, K, V, mask=None):
        # Q, K, V shape: [batch_size, n_heads, seq_len, head_dim]
        # mask shape: [batch_size, seq_len, seq_len]
        # Output shape: [batch_size, n_heads, seq_len, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output, attn_weights
    
    def split_heads(self, x, batch_size):
        # x shape: [batch_size, seq_len, d_model]
        # Output shape: [batch_size, n_heads, seq_len, head_dim]
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def combine_heads(self, x, batch_size):
        # x shape: [batch_size, n_heads, seq_len, head_dim]
        # Output shape: [batch_size, seq_len, d_model]
        x = x.permute(0, 2, 1, 3)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        attn_output, attn_weights = self.scaled_dot_product_attn(Q, K, V, mask)
        
        attn_output = self.combine_heads(attn_output, batch_size)
        attn_output = self.W_o(attn_output)
        
        return attn_output, attn_weights

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=768):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        
        self.position_wise_ff = PositionWiseFeedForward(d_model, d_ff)
        
    def forward(self, x, mask, iter):
        if iter != 0:
            x = self.layernorm(x)

        mask = mask.unsqueeze(1)
        context = self.multi_head_attn(x, x, x)[0]
        out = self.dropout(context) + x
        out = self.position_wise_ff(out)
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        #x = top_vecs * mask[:, :, None].float()
        # x = x + pos_emb^T
        x = top_vecs + pos_emb.transpose(0, 1)

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](x, mask, i)

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) #* mask.float()

        return sent_scores