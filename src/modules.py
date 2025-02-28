import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._C import device
from torch.nn.modules.activation import LeakyReLU
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence




class SpaceTimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model   = d_model
        
        div_term_space = (torch.arange(0, d_model, 4).float() * -(math.log(np.pi*2) / d_model**2)).exp() * (np.pi / 180)
        div_term_time  = (torch.arange(0, d_model, 2).float() * -(math.log(1000.0) / d_model)).exp()
        self.register_buffer('div_term_space', div_term_space)
        self.register_buffer('div_term_time', div_term_time)
        
        self.W_space = nn.Parameter(torch.ones(1, 1, d_model), requires_grad=True)
        nn.init.xavier_uniform_(self.W_space)            
        
    def forward(self, spacetime_x):
        space_encoding = torch.zeros(spacetime_x.size(0), spacetime_x.size(1), self.d_model, device=spacetime_x.device)
        time_encoding  = torch.zeros(spacetime_x.size(0), spacetime_x.size(1), self.d_model, device=spacetime_x.device)
        
        space_encoding[:, :, 0::4] = torch.cos(spacetime_x[:, :, [1]] * self.div_term_space) * torch.sin(spacetime_x[:, :, [0]] * self.div_term_space)
        space_encoding[:, :, 1::4] = torch.sin(spacetime_x[:, :, [1]] * self.div_term_space) / np.log(np.pi)**2
        space_encoding[:, :, 2::4] = torch.cos(spacetime_x[:, :, [1]] * self.div_term_space) * torch.cos(spacetime_x[:, :, [0]] * self.div_term_space)
        space_encoding[:, :, 3::4] = -torch.sin(spacetime_x[:, :,[1]] * self.div_term_space) / np.log(np.pi)**2
        
        time_encoding[:, :, 0::2]  = torch.sin(spacetime_x[:, :, [2]] * self.div_term_time)
        time_encoding[:, :, 1::2]  = torch.cos(spacetime_x[:, :, [2]] * self.div_term_time)
        
        space_encoding = space_encoding * self.W_space
                
        return space_encoding, time_encoding



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()

        encoding = torch.zeros(max_length, d_model).float()
        encoding.requires_grad = False

        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(100000.0) / d_model)).exp()

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        return self.encoding[:, :x.size(1)]



class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, query, key, value, mask=None, dropout=None):
        """
            query   : shape (batch_size, n_heads, max_len, d_q)
            key     : shape (batch_size, n_heads, max_len, d_k)
            value   : shape (batch_size, n_heads, max_len, d_v)
            mask    : shape (batch_size, 1, max_len, max_len)
            dropout : nn.Dropout
        """
        d_k = query.size(-1)
        # d_k = d_q = d_v = d_model // n_heads
        
        scores = torch.matmul((query / self.temperature), key.transpose(-2, -1))
        # query                 : (batch_size, n_heads, max_len, d_k)
        # key.transpose(-2, -1) : (batch_size, n_heads, d_k, max_len) --> scores : (batch_size, n_heads, max_len, max_len)

        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1) # (batch_size, n_heads, max_len, max_len[0:1])

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
        # p_attn : (batch_size, n_heads, max_len, max_len[0:1])
        # value  : (batch_size, n_heads, max_len, d_v) --> scaled value : (batch_size, n_heads, max_len, d_v)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, temperature, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        if d_k is not None:
            self.d_k = d_k
        else:
            assert d_model % n_heads == 0
            self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, (self.n_heads * self.d_k), bias=False)
        self.k_linear = nn.Linear(d_model, (self.n_heads * self.d_k), bias=False)
        self.v_linear = nn.Linear(d_model, (self.n_heads * self.d_k), bias=False)
        self.attn = ScaledDotProductAttention(temperature)
        self.dropout = nn.Dropout(p=dropout)

        self.out_linear = nn.Linear((self.d_k * self.n_heads), d_model, bias=False) # d_k * n_heads = d_model

    def forward(self, query, key, value, mask=None):
        """
            query : shape (batch_size, max_len, d_model)
            key   : shape (batch_size, max_len, d_model)
            value : shape (batch_size, max_len, d_model)
            mask  : shape (batch_size, max_len, max_len)        
        """
        batch_size = query.size(0)

        query = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key   = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # query, key, value : (batch_size, max_len, n_heads, d_k).transpose(1, 2) 
        #                     --> (batch_size, n_heads, max_len, d_k)

        scaled_value, p_attn = self.attn(query, key, value, mask=mask, dropout=self.dropout)
        # scaled_value : (batch_size, n_heads, max_len, d_v)
        # p_attn       : (batch_size, n_heads, max_len, max_len[0:1])

        scaled_value = scaled_value.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        # scaled_value.transpose(1, 2) : (batch_size, max_len, n_heads, d_v)
        #                                .view() --> (batch_size, max_len, d_model)

        return self.out_linear(scaled_value), p_attn
        # out : (batch_size, max_len, d_model)


        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
            x : (batch_size, max_len, d_model)
            activation = ReLU
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
        # out : (batch_size, max_len, d_model)
        
        

class MultiHeadChannelAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, in_channels, dropout, reduction_ratio=0.5):
        super(MultiHeadChannelAttention, self).__init__()
        self.in_channels     = in_channels
        self.reduction_ratio = reduction_ratio

        self.d_model = d_model
        if d_k is not None:
            self.d_k = d_k
        else:
            assert d_model % n_heads == 0
            self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=dropout)
        
        self.W_in = nn.Linear(self.d_model, (self.n_heads*self.d_k), bias=False)

        # Squeeze
        self.W1 = nn.Parameter(torch.ones(self.n_heads, 
                                          self.in_channels, 
                                          int(self.in_channels * float(self.reduction_ratio))))
        self.b1 = nn.Parameter(torch.ones(n_heads, 
                                          1, 
                                          int(self.in_channels * float(self.reduction_ratio))))
        # Exitation
        self.W2 = nn.Parameter(torch.ones(self.n_heads, 
                                          int(self.in_channels * float(self.reduction_ratio)), 
                                          self.in_channels))
        self.b2 = nn.Parameter(torch.ones(n_heads,
                                          1, 
                                          self.in_channels))
        # Init
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.b1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.b2)        

        self.W_out = nn.Linear((self.n_heads*self.d_k), self.d_model, bias=False)

    def forward(self, x):
        """
        x : (batch_size, max_len, channel=4, d_model)
        """
        x = self.W_in(x).view(-1, self.in_channels, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        # x : (batch_size*max_len, channel=5, n_heads, d_k).transpose(1, 2)
        #                   --> (batch_size*max_len, n_heads, channels, d_k)
        
        x = x.view(-1, (self.n_heads*self.in_channels), self.d_k)
        avg_pool = F.adaptive_avg_pool1d(x, 1).view(-1, self.n_heads, 1, self.in_channels)
        max_pool = F.adaptive_max_pool1d(x, 1).view(-1, self.n_heads, 1, self.in_channels)
        # x    : (batch_size*max_len, n_head*channels, d_k)
        # pool : (batch_size*max_len, n_head*channels, 1).view()
        #                   --> (batch_size*max_len, n_heads, 1, channels=5)

        # multi-head channel attention
        avg_pool_bck = torch.matmul(F.relu((torch.matmul(avg_pool, self.W1) + self.b1)), self.W2) + self.b2
        max_pool_bck = torch.matmul(F.relu((torch.matmul(max_pool, self.W1) + self.b1)), self.W2) + self.b2
        # pool_bck : (batch_size*max_len, n_head, 1, channels=5)

        pool_sum = avg_pool_bck + max_pool_bck
        attn_score = self.dropout(F.sigmoid(pool_sum).transpose(-1, -2).contiguous())
        # attn_score : (batch_size*max_len, n_heads, channels, 1)

        x = x.view(-1, self.n_heads, self.in_channels, self.d_k)
        x = x * attn_score
        # scaled_x : (batch_size*max_len, n_heads, channels, d_k)
        
        x = x.transpose(-1, -2).contiguous().view(-1, self.n_heads*self.d_k, self.in_channels)
        # scaled_x : (batch_size*max_len, n_heads*d_k, channels)

        out_pool = F.adaptive_max_pool1d(x, 1).transpose(-1, -2).contiguous()
        # out_pool : (batch_size*max_len, n_heads*d_k, 1) --> (batch_size*max_len, 1, n_heads*d_k)
        
        out_pool = self.W_out(out_pool)
        # x : (batch_size*max_len, 1, d_model)

        return out_pool, attn_score
        # out        : (batch_size*max_len, 1, d_model)
        # attn_score : (batch_size*max_len, n_heads, channels, 1)
        


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, d_k, temperature, dropout, block_no):
        super(EncoderLayer, self).__init__()

        self.block_no = block_no
        
        self.norm_chnl = nn.LayerNorm(d_model, eps=1e-6)
        self.chnl_attn = MultiHeadChannelAttention(d_model, n_heads, d_k, in_channels=4, dropout=0.0)
        self.drop_chnl = nn.Dropout(p=dropout)
        
        # self.norm_aggt = nn.LayerNorm(d_model, eps=1e-6)
        # self.aggregate = FeatureAggregation(d_model, n_heads, d_k, n_features, sub_length, dropout)
        # self.drop_aggt = nn.Dropout(p=dropout)

        self.norm_attn = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k, temperature, dropout)
        self.drop_attn = nn.Dropout(p=dropout)

        self.norm_ffn = nn.LayerNorm(d_model, eps=1e-6)
        self.feed_fwd = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.drop_ffn = nn.Dropout(p=dropout)

    def forward(self, x, attn_mask=None):
        """
        x : (batch_size, max_len, channel=4, d_model)
            channel [tkn_x, shp_x, dpt_x, ptn_x]
        """
        batch_size, max_length, n_channels, d_model = x.size()
        
        residual_x, agg_x = torch.split(x, [1, n_channels-1], dim=2)
        # residual_x : (batch_size, max_length, 1, d_model)
        # agg_x      : (batch_size, max_length, n_channels-1, d_model)

        x = x.view(-1, n_channels, d_model)
        x, p_chattn = self.chnl_attn(x)
        x        = x.view(batch_size, max_length, d_model)
        p_chattn = p_chattn.view(batch_size, max_length, -1, n_channels)
        x = residual_x.squeeze(2) + self.drop_chnl(x)
        x = self.norm_chnl(x)
        # x        : (batch_size, max_length, d_model)
        # p_chattn : (batch_size, max_length, n_heads, n_channels)
        
        # x = x.view(batch_size, max_length, n_channels, d_model)        
        # x, agg_x = torch.split(x, [1, x.size(2)-1], dim=2)
        # x        = x.squeeze(2)

        residual_x = x
        x, p_sfattn = self.self_attn(x, x, x, attn_mask)
        x = residual_x + self.drop_attn(x)
        x = self.norm_attn(x)
        
        x = torch.cat([x.unsqueeze(-2), agg_x], dim=2)
        
        residual_x = x
        x = self.feed_fwd(x)
        x = residual_x + self.drop_ffn(x)
        x = self.norm_ffn(x)
        
        return x, p_chattn, p_sfattn
        # x        : (batch_size, max_len, channel=4, d_model)
        # p_chattn : (batch_size, max_len, n_heads, channel=5)
        # p_sfattn : (batch_size, n_heads, max_len, max_len)
