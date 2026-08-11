import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec_CM import Encoder, EncoderLayer
from layers.SelfAttention_Family_CM import FullAttention, AttentionLayer, FlashAttention, ProbAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np
from utils.tools import standard_scaler
import os

def transform_corr(corr,s,b):
    scale = torch.exp(s)
    M = 1/(1+torch.exp(-scale*corr.to(s.device)+b))
    return M


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2402.02332
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.embed = nn.Linear(configs.seq_len, configs.d_model)
        self.backbone = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads) if configs.attn else None,
                    configs.d_model,
                    configs.d_block,
                    configs.d_ff,
                    dropout=configs.dropout,
                    gate = configs.gate
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if configs.d_block != configs.pred_len:
            self.align = nn.Linear(configs.d_block, configs.pred_len)
        else:
            self.align = nn.Identity()

        self.dname = configs.data_path.split('.csv')[0]
        if 'PEMS' in self.dname:
            corr = np.abs(np.load(os.path.join('corr',f'{self.dname.split(".npz")[0]}_corr.npy')))
        elif 'olar' in self.dname:
            corr = np.abs(np.load(os.path.join('corr','Solar_corr.npy')))
        elif self.dname == 'electricity':
            corr = np.abs(np.load(os.path.join('corr',f'ECL_corr.npy')))
        else:
            corr = np.abs(np.load(os.path.join('corr',f'{self.dname}_corr.npy')))
        
        
            
        self.corr = torch.FloatTensor(corr-corr.mean())#.to(configs.gpu)
        self.alpha_ds = nn.Parameter(torch.Tensor([1.0])*configs.init_alpha)#.to(configs.gpu)
        self.beta_ds = nn.Parameter(torch.Tensor([0.0]))#.to(configs.gpu)
        
        print('Minusformer ...')


    def forward(self, x, x_mark, x_dec, x_mark_dec, mask=None):
        cm = transform_corr(self.corr, self.alpha_ds, self.beta_ds)
        x = x.permute(0,2,1)
        scaler = standard_scaler(x)
        x = scaler.transform(x)
        if x_mark is not None:
            x_emb = self.embed(torch.cat((x, x_mark.permute(0,2,1)),1))  
        else:
            x_emb = self.embed(x)
        
        output = self.backbone(x_emb, cm)
        output = self.align(output)
        output = scaler.inverted(output[:, :x.size(1), :])
        return output.permute(0,2,1)
 
