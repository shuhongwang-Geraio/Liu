import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalSelectiveSSM(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Conv1d for local features
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        # SSM Parameters
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            repeats=self.d_inner,
            dim=0
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Dynamic parameter projection
        self.x_proj = nn.Linear(self.d_inner, (math.ceil(d_model / 16) + self.d_state * 2))
        self.dt_proj = nn.Linear(math.ceil(d_model / 16), self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def parallel_scan(self, u, delta, A, B, C):

        batch, seq_len, d_in = u.shape
        d_state = self.d_state
        
      
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        
   
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        
        x = torch.zeros(batch, d_in, d_state, device=u.device)
        ys = []
        
 
        for t in range(seq_len):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y = torch.einsum('bdn,bn->bd', x, C[:, t])
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
   
        self.encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU()
        )
      
        self.decoder = nn.Linear(dim, dim * 2)
        
      
        self.last_recon_loss = 0.0

    def forward(self, x):
     
        B, L, D = x.shape
        
       
        if L % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            L = L + 1
            
   
        #x_reshaped = x.view(B, L // 2, D * 2)
        x_reshaped = x.reshape(B, L // 2, D * 2)

    
        z = self.encoder(x_reshaped) # [B, L/2, D]
        
   
        if self.training:
            x_recon = self.decoder(z)
      
            self.last_recon_loss = F.mse_loss(x_recon, x_reshaped.detach())
        else:
            self.last_recon_loss = 0.0
            
        return z



class RenormalizationGate(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, 1) # Output single scalar per token (or per sequence)
        )
        
    def forward(self, local, global_up):
  
        logits = self.net(torch.cat([local, global_up], dim=-1))
        gate = torch.sigmoid(logits) # g in [0, 1]
        return gate


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
            g_current = gates_list[i]
            g_next = gates_list[i+1]
            
     
            loss += F.mse_loss(g_current, g_next.detach())
            
     
            
        return loss

    def forward_recursive(self, x, depth=0):
        B, L, D = x.shape
        

        time_scale = 2.0 ** depth
        local_feature = x + self.universal_ssm(self.norm(x), time_scale_factor=time_scale)
        
   
        if depth >= self.max_depth or L < self.min_len:
      
            return local_feature


        x_coarse = self.downsampler(local_feature)
        
    
        if self.training:
            self.aux_losses['recon'].append(self.downsampler.last_recon_loss)

 
        global_context = self.forward_recursive(x_coarse, depth + 1)
        
  
        global_context_t = global_context.transpose(1, 2)
        global_context_up = F.interpolate(
            global_context_t, size=L, mode='linear', align_corners=False
        ).transpose(1, 2)
        

        if self.training:
            consistency_loss = F.mse_loss(global_context_up, local_feature.detach())
            self.aux_losses['consistency'].append(consistency_loss)


        gate = self.gate_net(local_feature, global_context_up)

        if self.training:
            self.aux_losses['gates'].append(gate.mean())

        output = local_feature * (1 - gate) + global_context_up * gate
        return output

    def forward(self, x, return_aux_loss=False):

        self.aux_losses = {'recon': [], 'consistency': [], 'gates': []}
        

        output = self.forward_recursive(x)
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        if self.training and return_aux_loss:

            if self.aux_losses['recon']:
                total_aux_loss += sum(self.aux_losses['recon']) * 1.0
            

            if self.aux_losses['consistency']:
                total_aux_loss += sum(self.aux_losses['consistency']) * 1.0
                
  
            if self.aux_losses['gates']:

                gates_stack = torch.stack(self.aux_losses['gates'])
                total_aux_loss += self.compute_flow_loss(gates_stack) * 0.1
                
        if return_aux_loss:
            return output, total_aux_loss
        else:
            return output



if __name__ == "__main__":
    torch.manual_seed(42)
    
  
    BATCH = 2
    LENGTH = 128
    DIM = 32
    
 
    model = DynamicFractalMamba(d_model=DIM, max_depth=4, min_len=8)
    model.train()
    
    x = torch.randn(BATCH, LENGTH, DIM)
    
    print("-" * 50)
    print("Theoretical Fractal Mamba Initialized")
    print(f"Total Params: {sum(p.numel() for p in model.parameters())}")
    print("-" * 50)
    
  
    y, aux_loss = model(x, return_aux_loss=True)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
    print(f"Theoretical Constraints Loss: {aux_loss.item():.6f}")
    
 
    task_loss = y.sum()
    
 
    total_loss = task_loss + aux_loss
    total_loss.backward()
    
    print("-" * 50)
    print("Backward pass successful.")
    print("Gradients computed for Fixed Point, Flow, and Info Conservation constraints.")
    print("-" * 50)
