import os
import math
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import ot
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



def fix_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def permutation(feature: torch.Tensor):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    return feature[ids]



class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.f_k.weight.data)
        if self.f_k.bias is not None:
            self.f_k.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class Encoder_map(nn.Module):
    def __init__(self, n_cell, n_spot):
        super().__init__()
        self.M = Parameter(torch.FloatTensor(n_cell, n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)

    def forward(self):
        return self.M


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm["label_CSL"] = label_CSL


def get_feature(adata, deconvolution=False):
    if deconvolution:
        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata.var["highly_variable"]]

    if isinstance(adata_Vars.X, (csc_matrix, csr_matrix)):
        feat = adata_Vars.X.toarray()[:, :]
    else:
        feat = adata_Vars.X[:, :]

    feat_a = feat[np.random.permutation(feat.shape[0])]
    adata.obsm["feat"] = feat
    adata.obsm["feat_a"] = feat_a
    
   
    coords = adata.obsm['spatial'].copy()
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-6)
    adata.obsm["normalized_coords"] = coords


def construct_interaction(adata, n_neighbors=3):
    position = adata.obsm["spatial"]
    distance_matrix = ot.dist(position, position, metric="euclidean")
    n_spot = distance_matrix.shape[0]
    adata.obsm["distance_matrix"] = distance_matrix

    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm["graph_neigh"] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm["adj"] = adj


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm["spatial"]
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm["graph_neigh"] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm["adj"] = adj


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    return normalize_adj(adj) + np.eye(adj.shape[0])


def spatial_sort_index(adata):
    """
    Zigzag (Serpentine) Sorting to maintain physical continuity.
    """
    xy = adata.obsm["spatial"]
    df = pd.DataFrame(xy, columns=['x', 'y'])
    df['orig_index'] = np.arange(len(df))
    
    
    df['y_bin'] = pd.cut(df['y'], bins=100, labels=False) 
    
   
    df = df.sort_values(by=['y_bin', 'x'])
    

    groups = []
    for y_val, group in df.groupby('y_bin'):
        if y_val % 2 == 1:
            group = group.sort_values('x', ascending=False)
        else:
            group = group.sort_values('x', ascending=True)
        groups.append(group)
        
    df_sorted = pd.concat(groups)
    
    order = df_sorted['orig_index'].values
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return order, inv


class MinimalSelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            repeats=self.d_inner,
            dim=0,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.x_proj = nn.Linear(self.d_inner, (math.ceil(d_model / 16) + self.d_state * 2))
        self.dt_proj = nn.Linear(math.ceil(d_model / 16), self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def parallel_scan(self, u, delta, A, B, C):
        batch, seq_len, d_in = u.shape
        d_state = self.d_state
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        x = torch.zeros(batch, d_in, d_state, device=u.device)
        ys = []
        for t in range(seq_len):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y = torch.einsum("bdn,bn->bd", x, C[:, t])
            ys.append(y)
        return torch.stack(ys, dim=1)

    def forward(self, x, time_scale_factor=1.0):
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :seq_len]
        x_inner = self.act(x_inner).transpose(1, 2)

        dt_rank = math.ceil(self.d_model / 16)
        x_dbl = self.x_proj(x_inner)
        dt_raw, B, C = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        dt = dt * time_scale_factor

        A = -torch.exp(self.A_log)
        y = self.parallel_scan(x_inner, dt, A, B, C)

        y = y + x_inner * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class InformationPreservingDownsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Sequential(nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.SiLU())
        self.decoder = nn.Linear(dim, dim * 2)
        self.last_recon_loss = 0.0

    def forward(self, x):
        B, L, D = x.shape
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            L = L + 1
        x_reshaped = x.view(B, L // 2, D * 2)
        z = self.encoder(x_reshaped)
        if self.training:
            x_recon = self.decoder(z)
            self.last_recon_loss = F.mse_loss(x_recon, x_reshaped.detach())
        else:
            self.last_recon_loss = 0.0
        return z


class RenormalizationGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, local, global_up):
        logits = self.net(torch.cat([local, global_up], dim=-1))
        return torch.sigmoid(logits)


class DynamicFractalMamba(nn.Module):
    def __init__(self, d_model, max_depth=4, min_len=8):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        self.min_len = min_len

        self.universal_ssm = MinimalSelectiveSSM(d_model)
        self.downsampler = InformationPreservingDownsample(d_model)
        self.gate_net = RenormalizationGate(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.aux_losses = {}

    def compute_flow_loss(self, gates_list):
        if len(gates_list) < 2:
            return torch.tensor(0.0, device=gates_list[0].device)
        loss = 0.0
        for i in range(len(gates_list) - 1):
            loss += F.mse_loss(gates_list[i], gates_list[i + 1].detach())
        return loss

    def forward_recursive(self, x, depth=0):
        B, L, D = x.shape
        time_scale = 2.0 ** depth
        local_feature = x + self.universal_ssm(self.norm(x), time_scale_factor=time_scale)

        if depth >= self.max_depth or L < self.min_len:
            return local_feature

        x_coarse = self.downsampler(local_feature)
        if self.training:
            self.aux_losses["recon"].append(self.downsampler.last_recon_loss)

        global_context = self.forward_recursive(x_coarse, depth + 1)

        global_context_t = global_context.transpose(1, 2)
        global_context_up = F.interpolate(global_context_t, size=L, mode="linear", align_corners=False).transpose(1, 2)

        if self.training:
            consistency_loss = F.mse_loss(global_context_up, local_feature.detach())
            self.aux_losses["consistency"].append(consistency_loss)

        gate = self.gate_net(local_feature, global_context_up)
        if self.training:
            self.aux_losses["gates"].append(gate.mean())

        output = local_feature * (1 - gate) + global_context_up * gate
        return output

    def forward(self, x, return_aux_loss=False):
        self.aux_losses = {"recon": [], "consistency": [], "gates": []}
        out = self.forward_recursive(x)

        total_aux = torch.tensor(0.0, device=x.device)
        if self.training and return_aux_loss:
            if self.aux_losses["recon"]:
                total_aux = total_aux + sum(self.aux_losses["recon"]) * 1.0
            if self.aux_losses["consistency"]:
                total_aux = total_aux + sum(self.aux_losses["consistency"]) * 1.0
            if self.aux_losses["gates"]:
                gates_stack = torch.stack(self.aux_losses["gates"])
                total_aux = total_aux + self.compute_flow_loss(gates_stack) * 0.1

        return (out, total_aux) if return_aux_loss else out


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.x_embed = nn.Linear(1, d_model // 2)
        self.y_embed = nn.Linear(1, d_model // 2)
        self.act = nn.SiLU()

    def forward(self, x, coords):
        # coords: [Batch, Seq, 2]
        emb_x = self.x_embed(coords[:, :, 0:1])
        emb_y = self.y_embed(coords[:, :, 1:2])
        pos_emb = torch.cat([emb_x, emb_y], dim=-1)
        return x + self.act(pos_emb)


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(0.0)

    def forward(self, input, adj):
        # input: [Seq, Dim], adj: [Seq, Seq]
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class FractalMambaSTAutoEncoder(nn.Module):
    def __init__(self, num_genes: int, d_model: int = 128, max_depth: int = 4, min_len: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(num_genes, d_model)
        
        self.pos_encoder = SpatialPositionalEncoding(d_model)
        self.gcn_pre = GraphConv(d_model, d_model)
        
        self.backbone = DynamicFractalMamba(d_model=d_model, max_depth=max_depth, min_len=min_len)
        self.out_proj = nn.Linear(d_model, num_genes)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj, coords, return_aux_loss=False):
        h = self.in_proj(x)
        h = self.pos_encoder(h, coords)

        h_squeezed = h.squeeze(0)
        h_gcn = self.gcn_pre(h_squeezed, adj) 
        h = h.squeeze(0) + F.silu(h_gcn) 
        h = h.unsqueeze(0)
        
        if return_aux_loss:
            h2, aux = self.backbone(h, return_aux_loss=True)
            h2 = self.norm(h2)
            rec = self.out_proj(h2)
            return rec, h2, aux
        else:
            h2 = self.backbone(h, return_aux_loss=False)
            h2 = self.norm(h2)
            rec = self.out_proj(h2)
            return rec, h2


class Encoder(Module):
    def __init__(self, in_features: int, out_features: int, graph_neigh, dropout=0.0, act=F.relu,
                 d_model: int = 128, max_depth: int = 4, min_len: int = 8, aux_loss_weight: float = 0.1):
        super().__init__()
        self.ae = FractalMambaSTAutoEncoder(num_genes=in_features, d_model=d_model, max_depth=max_depth, min_len=min_len)
        self.aux_loss_weight = aux_loss_weight
        self.disc = Discriminator(d_model) 
        self.last_aux_loss = torch.tensor(0.0)
        self.sigm = nn.Sigmoid()

    def forward(self, feat: torch.Tensor, feat_a: torch.Tensor, adj: torch.Tensor, coords: torch.Tensor):
        feat_b = feat.unsqueeze(0)
        coords_b = coords.unsqueeze(0)
        self.last_aux_loss = torch.tensor(0.0, device=feat.device)

        if self.training:
            rec_b, emb_b, aux = self.ae(feat_b, adj, coords_b, return_aux_loss=True)
            self.last_aux_loss = aux * self.aux_loss_weight
        else:
            rec_b, emb_b = self.ae(feat_b, adj, coords_b, return_aux_loss=False)

        reconstructed_feat = rec_b.squeeze(0)
        hidden_embeddings = emb_b.squeeze(0)
        
        feat_a_b = feat_a.unsqueeze(0)
        _, emb_a_b = self.ae(feat_a_b, adj, coords_b, return_aux_loss=False)
        emb_a = emb_a_b.squeeze(0)

        c = self.sigm(hidden_embeddings.mean(dim=0)).unsqueeze(0)
        logits = self.disc(c, hidden_embeddings, emb_a)
        
        return hidden_embeddings, reconstructed_feat, logits, logits


class Encoder_sc(nn.Module):
    def __init__(self, dim_input: int, dim_output: int, d_model: int = 128, max_depth: int = 4, min_len: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(dim_input, d_model)
        self.backbone = DynamicFractalMamba(d_model=d_model, max_depth=max_depth, min_len=min_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x_b = x.unsqueeze(0)
        h = self.in_proj(x_b)
        h2 = self.backbone(h, return_aux_loss=False)
        h2 = self.norm(h2)
        return h2.squeeze(0)


Encoder_sparse = Encoder


class FractalMambaST():
    def __init__(
        self,
        adata,
        adata_sc=None,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        learning_rate_sc=1e-2,
        weight_decay=0.0,
        epochs=3000,
        dim_output=32,
        random_seed=41,
        alpha=30,
        beta=1, 
        theta=0.1,
        lamda1=10,
        lamda2=1,
        deconvolution=False,
        datatype="10X",
        d_model=128,
        max_depth=4,
        min_len=8,
        aux_loss_weight=0.1,
    ):
        self.adata = adata.copy()
        self.adata_sc = adata_sc.copy() if adata_sc is not None else None
        self.device = device

        self.learning_rate = learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2

        self.deconvolution = deconvolution
        self.datatype = datatype

        self.d_model = d_model
        self.max_depth = max_depth
        self.min_len = min_len
        self.aux_loss_weight = aux_loss_weight

        fix_seed(self.random_seed)

        if "highly_variable" not in self.adata.var.keys():
            preprocess(self.adata)

        if "adj" not in self.adata.obsm.keys():
            if self.datatype in ["Stereo", "Slide"]:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if "label_CSL" not in self.adata.obsm.keys():
            add_contrastive_label(self.adata)

        if "feat" not in self.adata.obsm.keys():
            get_feature(self.adata)

        order, inv = spatial_sort_index(self.adata)
        self.order = order
        self.inv_order = inv

        feat_np = self.adata.obsm["feat"].copy()[self.order]
        feat_a_np = self.adata.obsm["feat_a"].copy()[self.order]
        coords_np = self.adata.obsm["normalized_coords"].copy()[self.order] 
        
        adj_np = preprocess_adj(self.adata.obsm["adj"])
        adj_np = adj_np[self.order, :][:, self.order]

        self.features = torch.FloatTensor(feat_np).to(self.device)
        self.features_a = torch.FloatTensor(feat_a_np).to(self.device)
        self.coords = torch.FloatTensor(coords_np).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm["label_CSL"]).to(self.device)
        
        self.adj_t = torch.FloatTensor(adj_np).to(self.device)
        self.graph_neigh = torch.FloatTensor(self.adata.obsm["graph_neigh"].copy() + np.eye(self.adata.n_obs)).to(self.device)

        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output

    def train(self):
        self.model = Encoder(
            in_features=self.dim_input,
            out_features=self.dim_output,
            graph_neigh=self.graph_neigh,
            d_model=self.d_model,
            max_depth=self.max_depth,
            min_len=self.min_len,
            aux_loss_weight=self.aux_loss_weight,
        ).to(self.device)

        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        print("Begin to train ST data (FractalMamba encoder)...")
        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.features_a = permutation(self.features)

            hiden_feat, emb_rec_sorted, ret, ret_a = self.model(
                self.features, self.features_a, self.adj_t, self.coords
            )

            loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            loss_feat = F.mse_loss(self.features, emb_rec_sorted)

            aux_loss = getattr(self.model, "last_aux_loss", torch.tensor(0.0, device=self.device))
            
            loss = self.alpha * loss_feat + self.beta * (loss_sl_1 + loss_sl_2) + aux_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished for ST data!")

        with torch.no_grad():
            self.model.eval()
            emb_rec_sorted = self.model(
                self.features, self.features_a, self.adj_t, self.coords
            )[0]
            
            emb_rec = emb_rec_sorted.detach().cpu().numpy()[self.inv_order]

            if self.deconvolution:
                self.emb_rec = torch.FloatTensor(emb_rec).to(self.device)
                return self.emb_rec

            self.emb_rec = emb_rec
            self.adata.obsm["emb"] = self.emb_rec
            return self.adata

    def train_sc(self):
        self.model_sc = Encoder_sc(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            d_model=self.d_model,
            max_depth=self.max_depth,
            min_len=self.min_len,
        ).to(self.device)

        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)

        print("Begin to train scRNA data (FractalMamba AE)...")
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            rec = self.model_sc(self.feat_sc)
            loss = F.mse_loss(rec, self.feat_sc)
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()

        print("Optimization finished for cell representation learning!")
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
            return emb_sc
            
    def cosine_similarity(self, pred_sp, emb_sp):
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        if torch.any(torch.isnan(M)):
            M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)
        return M

    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        mat = self.cosine_similarity(pred_sp, emb_sp)
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        ave = torch.div(p, k)
        loss = -torch.log(ave).mean()
        return loss

    def loss_map(self, emb_sp, emb_sc, map_matrix):
        map_probs = F.softmax(map_matrix, dim=1)
        pred_sp = torch.matmul(map_probs.t(), emb_sc)
        loss_recon = F.mse_loss(pred_sp, emb_sp, reduction="mean")
        loss_NCE = self.Noise_Cross_Entropy(pred_sp, emb_sp)
        return loss_recon, loss_NCE, pred_sp

    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()

        self.adata.obsm["emb_sp"] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm["emb_sc"] = emb_sc.detach().cpu().numpy()

        emb_sp_n = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc_n = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)

        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)
        optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        print("Begin to learn mapping matrix...")
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            map_matrix = self.model_map()

            loss_recon, loss_NCE, _ = self.loss_map(emb_sp_n, emb_sc_n, map_matrix)
            loss = self.lamda1 * loss_recon + self.lamda2 * loss_NCE

            optimizer_map.zero_grad()
            loss.backward()
            optimizer_map.step()

        print("Mapping matrix learning finished!")

        with torch.no_grad():
            self.model_map.eval()
            map_matrix = F.softmax(self.model_map(), dim=1).cpu().numpy()
            self.adata.obsm["map_matrix"] = map_matrix.T
            return self.adata, self.adata_sc


def refine_label(adata, radius=50, key="label"):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm["spatial"]
    distance = ot.dist(position, position, metric="euclidean")
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    return [str(i) for i in list(new_type)]


def search_res(adata, n_clusters, method="leiden", use_rep="emb_pca", start=0.1, end=3.0, increment=0.01):
    print("Searching resolution...")
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())

        print(f"resolution={res}, cluster number={count_unique}")
        if count_unique == n_clusters:
            label = 1
            break
    assert label == 1, "Resolution not found. Try bigger range or smaller step."
    return res


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    rmclust = robjects.r["Mclust"]
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs["mclust"] = mclust_res.astype("int").astype("category")
    return adata


def clustering(adata, n_clusters=7, radius=50, key="emb", method="mclust",
               start=0.1, end=3.0, increment=0.01, refinement=False):
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm["emb_pca"] = embedding

    if method == "mclust":
        adata = mclust_R(adata, used_obsm="emb_pca", num_cluster=n_clusters)
        adata.obs["domain"] = adata.obs["mclust"]
    elif method == "leiden":
        res = search_res(adata, n_clusters, method=method, use_rep="emb_pca", start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs["domain"] = adata.obs["leiden"]
    elif method == "louvain":
        res = search_res(adata, n_clusters, method=method, use_rep="emb_pca", start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs["domain"] = adata.obs["louvain"]

    if refinement:
        new_type = refine_label(adata, radius, key="domain")
        adata.obs["domain"] = new_type


def try_attach_ground_truth(adata, file_fold):
    """
    Attempts to load metadata.tsv and attach Ground Truth to adata.obs['ground_truth'].
    """
    meta_path = os.path.join(file_fold, "metadata.tsv")
    if not os.path.exists(meta_path):
        print(f"Metadata file not found at {meta_path}. Skipping GT.")
        return None

    df_meta = pd.read_csv(meta_path, sep="\t")
    

    gt_candidates = ["ground_truth", "layer_guess", "annotation", "Group", "param", "celltype"]
    gt_col = next((c for c in gt_candidates if c in df_meta.columns), None)
    
    if gt_col:
        print(f"Found Ground Truth column: '{gt_col}'")
        # Ensure indices match if possible, otherwise assume order match
        if len(df_meta) == adata.n_obs:
            adata.obs["ground_truth"] = df_meta[gt_col].values
            # Filter out NaNs if any
            valid_mask = ~pd.isnull(adata.obs["ground_truth"])
            if not valid_mask.all():
                print(f"Warning: {np.sum(~valid_mask)} spots have NaN ground truth.")
        else:
            print("Warning: Metadata rows do not match adata observations.")
    else:
        print("No recognized Ground Truth column found in metadata.")


def compute_metrics_if_possible(adata):
    if "ground_truth" not in adata.obs.columns:
        print("No 'ground_truth' column in adata.obs. Skipping metrics.")
        return None

    # Filter out spots where GT is NaN
    adata2 = adata[~pd.isnull(adata.obs["ground_truth"])].copy()
    if adata2.n_obs == 0:
        return None

    y_pred = adata2.obs["domain"].astype(str)
    y_true = adata2.obs["ground_truth"].astype(str)

    ARI = metrics.adjusted_rand_score(y_pred, y_true)
    NMI = metrics.normalized_mutual_info_score(y_pred, y_true)
    AMI = metrics.adjusted_mutual_info_score(y_pred, y_true)
    HOM = metrics.homogeneity_score(y_true, y_pred)
    COM = metrics.completeness_score(y_true, y_pred)
    VME = metrics.v_measure_score(y_true, y_pred)

    print("-" * 30)
    print("CLUSTERING METRICS:")
    print("-" * 30)
    print(f"ARI: {ARI:.4f}")
    print(f"NMI: {NMI:.4f}")
    print(f"AMI: {AMI:.4f}")
    print(f"Homogeneity: {HOM:.4f}")
    print(f"Completeness: {COM:.4f}")
    print(f"V-measure: {VME:.4f}")
    print("-" * 30)

    return {
        "ARI": ARI,
        "NMI": NMI,
        "AMI": AMI,
        "Homogeneity": HOM,
        "Completeness": COM,
        "V-measure": VME,
    }
