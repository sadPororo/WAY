import math
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

from modules import *



class WAY(nn.Module):
    def __init__(self, args):
        super(WAY, self).__init__()

        # meta config
        self.vocab_size  = args.vocab_size
        self.n_class     = args.n_class
        self.n_ship      = args.n_ship
        self.max_length  = args.max_length
        self.padding_idx = args.padding_idx
        self.cls_idx     = args.cls_idx

        # model config
        self.n_blocks    = args.n_blocks
        self.d_model     = args.d_model
        self.d_ffn       = args.d_ffn
        self.n_heads     = args.n_heads
        self.d_k         = args.d_k
        self.temperature = args.temperature
        self.n_features  = args.n_features
        self.sub_length  = args.sub_length
        self.dropout     = args.dropout

        # embeddings
        assert self.d_model % 4 == 0
        self.spacetimeEncoding = SpaceTimeEncoding(self.d_model)
        self.classEmbedding    = nn.Embedding(self.n_class,    self.d_model, padding_idx=self.padding_idx)
        self.shipEmbedding     = nn.Embedding(self.n_ship,     self.d_model)
        self.patternEmbedding  = getattr(nn, 'GRU')(input_size   =self.n_features, 
                                                    hidden_size  =self.d_model, 
                                                    num_layers   =(self.n_blocks//2), 
                                                    batch_first  =True, 
                                                    bidirectional=False)

        self.encoder = nn.ModuleList(
            [EncoderLayer(self.d_model, 
                          self.d_ffn, 
                          self.n_heads,
                          self.d_k,
                          self.temperature,
                          self.dropout, i) for i in range(self.n_blocks)])

    def set_device(self, device):
        self.device = device

    def forward(self, spacetime_x, shiptype, depart, subseq_x, subseq_length):
        """
            spacetime_x   : (batch_size, max_len, 3)
            sent          : (batch_size, max_len)
            depart        : (batch_size,)
            subseq_x      : (batch_size, max_len, sub_len, n_features)
            subseq_length : (batch_size, max_len)
        """
        batch_size, max_length = subseq_length.size()
        
        pad_mask  = subseq_length.eq(self.padding_idx).unsqueeze(1).repeat(1, max_length, 1)
        seq_mask  = torch.ones(batch_size, max_length, max_length).triu(diagonal=1)
        attn_mask = ~torch.gt((pad_mask.to(dtype=seq_mask.dtype)+seq_mask), 0).unsqueeze(1).to(self.device)
        
        tkn_x, time_encoding = self.spacetimeEncoding(spacetime_x)
        dpt_x = self.classEmbedding(depart).unsqueeze(1).repeat(1, max_length, 1)
        shp_x = self.shipEmbedding(shiptype).unsqueeze(1).repeat(1, max_length, 1)
        # tkn_x : (batch_size, max_len, d_model//4)
        # shp_x : (batch_size, max_len, d_model//4)
        # dpt_x : (batch_size, max_len, d_model//4)
        
        subseq_x      = subseq_x.view(-1, self.sub_length, self.n_features)
        subseq_length = subseq_length.view(-1)

        pattern_in  = pack_padded_sequence(subseq_x[subseq_length.bool()], subseq_length[subseq_length.bool()], batch_first=True, enforce_sorted=False)
        pattern_out, hidden = self.patternEmbedding(pattern_in)
        pattern_out, outlen = pad_packed_sequence(pattern_out, batch_first=True, total_length=self.sub_length)
        ptn_x = torch.zeros((batch_size*max_length, self.patternEmbedding.hidden_size), device=self.device)
        ptn_x[subseq_length.bool()] = pattern_out[range(pattern_out.size(0)), subseq_length[subseq_length.bool()]-1].contiguous() # (batch_size*max_len, d_model)
        ptn_x = ptn_x.view(batch_size, -1, self.patternEmbedding.hidden_size) # (batch_size, max_len, d_model)

        x = torch.cat([tkn_x.unsqueeze(-2), shp_x.unsqueeze(-2), dpt_x.unsqueeze(-2), ptn_x.unsqueeze(-2)], dim=-2) + time_encoding.unsqueeze(-2)
        # x : (batch_size, max_len, d_model)
        
        chattn_list, sfattn_list = [], []
        for layer in self.encoder:
            x, p_chattn, p_sfattn = layer.forward(x, attn_mask)
            chattn_list.append(p_chattn)
            sfattn_list.append(p_sfattn)
            # x        : (batch_size, max_len, d_model)
            # p_sfattn : (batch_size, n_heads, max_len, max_len)
                    
        return x[:, :, 0], torch.stack(chattn_list), torch.stack(sfattn_list)
        # x           : (batch_size, max_len, d_model)
        # sfattn_list : (n_blocks, batch_size, n_heads, max_len, max_len)        



class DownstreamHead(nn.Module):
    def __init__(self, backbone):
        super(DownstreamHead, self).__init__()
        self.backbone = backbone

        self.linearLTD = nn.Linear(self.backbone.d_model, self.backbone.n_class)
        self.linearSTD = nn.Linear(self.backbone.d_model, self.backbone.vocab_size)
        self.linearCRD = nn.Linear(self.backbone.d_model, 2)
        self.linearEDA = nn.Linear(self.backbone.d_model, 1)
        self.linearETA = nn.Linear(self.backbone.d_model, 1)

    def set_device(self, device):
        print(f'   ::setting model device to {device}...')
        self.backbone.set_device(device)

    def forward(self, spacetime_x, shiptype, depart, subseq_x, subseq_length):
        """
            spacetime_x   : (batch_size, max_len, 3)        
            sent          : (batch_size, max_len)
            subseq_length : (batch_size, max_len)
            depart        : (batch_size,)
            subseq_x      : (batch_size, max_len, sub_len, n_features)
            subseq_length : (batch_size, max_len)
        """
        x, chattn_list, sfattn_list = self.backbone(spacetime_x, shiptype, depart, subseq_x, subseq_length)
        # x           : (batch_size, max_len, d_model)
        # attn_list   : (n_blocks, batch_size, n_heads, max_len, max_len)

        LTD_logit = self.linearLTD(x)
        STD_logit = self.linearSTD(x)
        CRD_logit = self.linearCRD(x)
        EDA_logit = self.linearEDA(x)
        ETA_logit = self.linearETA(x)
        # LTD_logit : (batch_size, max_len, n_class)
        # STD_logit : (batch_size, max_len, vocab_size)
        # CRD_logit : (batch_size, max_len, 2)
        # EDA_logit : (batch_size, max_len)
        # ETA_logit : (batch_size, max_len)
        
        return (LTD_logit, 
                STD_logit, 
                CRD_logit, 
                F.relu(EDA_logit).unsqueeze(-1), 
                F.relu(ETA_logit).unsqueeze(-1),
                chattn_list, 
                sfattn_list)
        # LTD_logit : (batch_size, max_len, n_class)
        # STD_logit : (batch_size, max_len, vocab_size)
        # CRD_logit : (batch_size, max_len, 2)
        # ETA_logit : (batch_size, max_len, 1)
        # sfattn_list : (n_blocks, batch_size, n_heads, max_len, max_len)

